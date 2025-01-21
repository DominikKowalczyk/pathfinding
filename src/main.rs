use piston_window::*;
use ::image::{DynamicImage, GrayImage, io::Reader as ImageReader};
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use clap::Parser;
use log::{info, debug, error};
use anyhow::{Result, Error};

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
    debug!("Heuristic from {:?} to {:?}: {:.2}", current, goal, h);
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
    debug!("Effort from {} to {} over distance {:.2}: {:.2}", from, to, distance, effort);
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
    visited_nodes: &mut Vec<(u32, u32)>,
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

    debug!("Starting pathfinding from {:?} to {:?}", start, goal);

    while let Some(Node { cost, position, .. }) = heap.pop() {
        debug!("Visiting node: {:?} with cost: {:.2}", position, cost);

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
            debug!("Path found: {:?} with total cost: {:.2}", path, cost);
            return Ok((cost, path));
        }

        if visited.contains(&position) {
            debug!("Node {:?} already visited, skipping", position);
            continue;
        }

        visited.insert(position);
        visited_nodes.push(position);

        for neighbor in neighbors(position, width, height) {
            let (nx, ny) = neighbor;
            let distance = if nx != position.0 && ny != position.1 {
                1.414 // Diagonal movement
            } else {
                1.0 // Horizontal/Vertical movement
            };
            let new_cost = cost
                + effort(
                    heightmap.get_pixel(position.0, position.1)[0],
                    heightmap.get_pixel(nx, ny)[0],
                    distance,
                    weight,
                    fatigue_factor,
                    temperature,
                );

            debug!("Effort from {:?} to {:?} over distance {:.2}: {:.2}", position, neighbor, distance, new_cost);

            if new_cost < dist[nx as usize][ny as usize] {
                dist[nx as usize][ny as usize] = new_cost;
                prev[nx as usize][ny as usize] = Some(position);
                heap.push(Node {
                    cost: new_cost,
                    priority: new_cost + heuristic((nx, ny), goal),
                    position: (nx, ny),
                });
                debug!("Updating node {:?} with new cost: {:.2} and priority: {:.2}", (nx, ny), new_cost, new_cost + heuristic((nx, ny), goal));
            }
        }

        debug!("Current state of priority queue: {:?}", heap);
    }

    error!("No valid path found.");
    Err(Error::msg("No valid path found"))
}

/// Function to draw the heightmap and pathfinding progress.
fn draw_scene(
    gray_img: &GrayImage,
    visited_nodes: &Vec<(u32, u32)>,
    current_path: &Vec<(u32, u32)>,
    c: &Context,
    g: &mut G2d,
) {
    // Draw the heightmap
    for (x, y, pixel) in gray_img.enumerate_pixels() {
        let brightness = pixel[0] as f32 / 255.0;
        rectangle(
            [brightness, brightness, brightness, 1.0],
            [x as f64, y as f64, 1.0, 1.0],
            c.transform,
            g,
        );
    }

    // Draw visited nodes
    for &(x, y) in visited_nodes {
        rectangle(
            [0.0, 1.0, 1.0, 1.0],
            [x as f64, y as f64, 1.0, 1.0],
            c.transform,
            g,
        );
    }

    // Draw current path
    for &(x, y) in current_path {
        rectangle(
            [1.0, 0.0, 0.0, 1.0],
            [x as f64, y as f64, 1.0, 1.0],
            c.transform,
            g,
        );
    }
}

#[derive(Parser)]
struct Args {
    #[clap(short, long, default_value = "70.0")]
    weight: f64,
    #[clap(short, long, default_value = "0.1")]
    fatigue_factor: f64,
    #[clap(short, long, default_value = "20.0")]
    temperature: f64,
}

fn main() -> Result<(), Error> {
    // Initialize logging
    env_logger::init();

    let args = Args::parse();

    // Load heightmap as a dynamic image
    info!("Loading heightmap image...");
    let img = ImageReader::open("heightmap.png")
        .map_err(|e| anyhow::Error::msg(format!("Failed to open heightmap.png: {}", e)))?
        .decode()
        .map_err(|e| anyhow::Error::msg(format!("Failed to decode heightmap image: {}", e)))?;

    // Validate that the image is grayscale
    info!("Validating grayscale image...");
    validate_grayscale_image(&img)?;

    // Convert to grayscale image
    let gray_img = img.to_luma8();
    let start = (0, 0);
    let goal = (gray_img.width() - 1, gray_img.height() - 1);

    let mut visited_nodes = Vec::new();
    let mut current_path = Vec::new();

    let mut window: PistonWindow = WindowSettings::new("Pathfinding Visualization", [800, 800])
        .exit_on_esc(true)
        .build()
        .map_err(|e| anyhow::Error::msg(format!("Failed to create window: {}", e)))?;

    while let Some(event) = window.next() {
        window.draw_2d(&event, |c, g, _| {
            clear([1.0; 4], g);
            draw_scene(&gray_img, &visited_nodes, &current_path, &c, g);
        });

        if !current_path.is_empty() {
            continue;
        }

        info!("Starting pathfinding...");
        match find_optimal_path(
            &gray_img,
            start,
            goal,
            args.weight,
            args.fatigue_factor,
            args.temperature,
            &mut visited_nodes,
            &mut current_path,
        ) {
            Ok((cost, _path)) => {
                info!("Pathfinding complete. Total cost: {}", cost);
            }
            Err(e) => {
                error!("Pathfinding failed: {:?}", e);
            }
        }
    }

    Ok(())
}
