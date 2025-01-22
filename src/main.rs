use ::image::{DynamicImage, GrayImage, io::Reader as ImageReader};
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use clap::Parser;
use log::{info, debug, error};
use anyhow::{Result, Error};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder, Window},
};

// Helper macro to add line number information to debug messages
macro_rules! debug_line {
    ($($arg:tt)*) => (
        debug!("{}: {}", line!(), format_args!($($arg)*))
    );
}

#[derive(Copy, Clone, PartialEq)]
#[derive(Debug)]
struct Node {
    cost: f64,     // g(n): the cost to reach this node
    priority: f64, // f(n) = g(n) + h(n): total cost (includes heuristic)
    position: (u32, u32),
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal) // Reverse order for min-heap
    }
}

/// Heuristic for A* pathfinding: Manhattan distance
fn heuristic(current: (u32, u32), goal: (u32, u32)) -> f64 {
    let h = (current.0 as f64 - goal.0 as f64).abs() + (current.1 as f64 - goal.1 as f64).abs();
    debug_line!("Heuristic from {:?} to {:?}: {:.2}", current, goal, h);
    h
}

/// Validates that the image is a grayscale image.
fn validate_grayscale_image(img: &DynamicImage) -> Result<(), Error> {
    match img {
        DynamicImage::ImageLuma8(_) => Ok(()),
        DynamicImage::ImageLuma16(_) => Ok(()),
        _ => Err(Error::msg(format!(
            "Heightmap must be a grayscale image. Found: {:?}",
            img
        ))),
    }
}

/// Calculates the effort required to move between two points, considering terrain factors.
fn effort(
    from: u8,
    to: u8,
    distance: f64,
    weight: f64,
    fatigue_factor: f64,
    temperature: f64,
) -> f64 {
    let height_diff = (to as f64 - from as f64).abs();
    let slope = height_diff / distance;
    let altitude = (from as f64 + to as f64) / 2.0; // Approximate altitude
    let effort = distance + slope_penalty(slope, weight, altitude, fatigue_factor, temperature);
    debug_line!("Effort from {} to {} over distance {:.2}: {:.2}", from, to, distance, effort);
    effort
}

/// Applies a penalty based on slope steepness and real-world effort factors.
fn slope_penalty(
    slope: f64,
    weight: f64,
    altitude: f64,
    fatigue_factor: f64,
    temperature: f64,
) -> f64 {
    let gravity_factor = 9.8 * weight;
    let altitude_penalty = if altitude > 2000.0 { 1.2 } else { 1.0 };
    let fatigue_penalty = 1.0 + fatigue_factor.min(0.5);
    let temp_penalty = if temperature < 0.0 || temperature > 35.0 {
        1.1
    } else {
        1.0
    };
    slope * gravity_factor * altitude_penalty * fatigue_penalty * temp_penalty
}

/// Generates valid neighbors of a given position.
fn neighbors(position: (u32, u32), width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    let (x, y) = position;
    if x > 0 { result.push((x - 1, y)); }
    if x < width - 1 { result.push((x + 1, y)); }
    if y > 0 { result.push((x, y - 1)); }
    if y < height - 1 { result.push((x, y + 1)); }
    result
}

/// Finds the optimal path between two points on a heightmap using A* algorithm.
fn find_optimal_path(
    heightmap: &GrayImage,
    start: (u32, u32),
    goal: (u32, u32),
    weight: f64,
    fatigue_factor: f64,
    temperature: f64,
    visited_nodes: &mut HashSet<(u32, u32)>,
    current_path: &mut Vec<(u32, u32)>,
) -> Result<(f64, Vec<(u32, u32)>), Error> {
    let (width, height) = heightmap.dimensions();
    let mut dist = vec![vec![f64::INFINITY; height as usize]; width as usize];
    let mut prev = vec![vec![None; height as usize]; width as usize];
    let mut visited = HashSet::new();
    let mut heap = BinaryHeap::new();

    dist[start.0 as usize][start.1 as usize] = 0.0;
    heap.push(Node {
        cost: 0.0,
        priority: heuristic(start, goal),
        position: start,
    });

    debug_line!("Starting pathfinding from {:?} to {:?}", start, goal);

    while let Some(Node { cost, position, .. }) = heap.pop() {
        debug_line!("Visiting node: {:?} with cost: {:.2}", position, cost);

        if position == goal {
            let mut path = Vec::new();
            let mut current = position;
            while let Some(p) = prev[current.0 as usize][current.1 as usize] {
                path.push(current);
                current = p;
            }
            path.push(start);
            path.reverse();
            *current_path = path.clone();
            debug_line!("Path found: {:?} with total cost: {:.2}", path, cost);
            return Ok((cost, path));
        }

        if visited.contains(&position) {
            debug_line!("Node {:?} already visited, skipping", position);
            continue;
        }

        visited.insert(position);
        visited_nodes.insert(position);

        for neighbor in neighbors(position, width, height) {
            let (nx, ny) = neighbor;
            let distance = if nx != position.0 && ny != position.1 {
                1.414 // Diagonal movement
            } else {
                1.0 // Horizontal/Vertical movement
            };
            let from_height = heightmap.get_pixel(position.0, position.1)[0];
            let to_height = heightmap.get_pixel(nx, ny)[0];
            debug_line!("Calculating effort from height {} to height {} over distance {:.2}", from_height, to_height, distance);
            let new_cost = cost
                + effort(
                    from_height,
                    to_height,
                    distance,
                    weight,
                    fatigue_factor,
                    temperature,
                );

            debug_line!("Effort from {:?} to {:?} over distance {:.2}: {:.2}", position, neighbor, distance, new_cost);

            if new_cost < dist[nx as usize][ny as usize] {
                dist[nx as usize][ny as usize] = new_cost;
                prev[nx as usize][ny as usize] = Some(position);
                heap.push(Node {
                    cost: new_cost,
                    priority: new_cost + heuristic((nx, ny), goal),
                    position: (nx, ny),
                });
                debug_line!("Updating node {:?} with new cost: {:.2} and priority: {:.2}", (nx, ny), new_cost, new_cost + heuristic((nx, ny), goal));
            }
        }

        debug_line!("Current state of priority queue: {:?}", heap);
    }

    error!("No valid path found.");
    Err(Error::msg("No valid path found"))
}

#[derive(Parser)]
struct Args {
    #[clap(short, long, default_value = "70.0")]
    weight: f64,

    #[clap(short, long, default_value = "0.2")]
    fatigue_factor: f64,

    #[clap(short, long, default_value = "20.0")]
    temperature: f64,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    texture_bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl State {
    async fn new(window: &Window, gray_img: &GrayImage) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let sc_desc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &sc_desc);

        let texture_size = wgpu::Extent3d {
            width: gray_img.width(),
            height: gray_img.height(),
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Heightmap Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("texture_bind_group"),
        });

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: sc_desc.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_data = [
            // positions    // tex_coords
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ];

        let index_data = [
            0u16, 1, 2,
            0, 2, 3,
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = index_data.len() as u32;

        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain: device.create_swap_chain(&surface, &sc_desc),
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            texture_bind_group,
            texture,
            texture_view,
            sampler,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.surface.configure(&self.device, &self.sc_desc);
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.swap_chain.get_current_frame()?.output;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame.view,
                    view: &view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
        frame.present();
        Ok(())
}
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    env_logger::init();

    let _args = Args::parse();

    let img = ImageReader::open("heightmap.png")?.decode()?;
    validate_grayscale_image(&img)?;

    let gray_img = img.to_luma8();
    let (width, height) = gray_img.dimensions();
    let start = (0, 0);
    let goal = (width - 1, height - 1);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Pathfinding Visualization")
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .build(&event_loop)?;

    let mut state = State::new(&window, &gray_img).await;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}