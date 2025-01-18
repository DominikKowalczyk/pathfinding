use image::{DynamicImage, GrayImage, ColorType, ImageBuffer, Luma, io::Reader as ImageReader};
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use std::env;

/// Represents a node in the priority queue for A* algorithm.
#[derive(Copy, Clone)]
struct Node {
    cost: f64,       // g(n): the cost to reach this node
    priority: f64,   // f(n) = g(n) + h(n): total cost (includes heuristic)
    position: (u32, u32),
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
    }
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.partial_cmp(&other.priority).unwrap_or(Ordering::Equal) // Priority comparison (f64)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other)) // Use `cmp` for `PartialOrd`
    }
}

/// Heuristic for A* pathfinding: Manhattan distance
fn heuristic(current: (u32, u32), goal: (u32, u32)) -> f64 {
    (current.0 as f64 - goal.0 as f64).abs() + (current.1 as f64 - goal.1 as f64).abs()
}

/// Validates that the image is a grayscale image.
fn validate_grayscale_image(img: &DynamicImage) -> Result<(), String> {
    if img.color() == ColorType::L8 || img.color() == ColorType::L16 {
        Ok(())
    } else {
        Err("Heightmap must be a grayscale image.".to_string())
    }
}

/// Calculates the effort required to move between two points, considering terrain factors.
fn effort(from: u8, to: u8, distance: f64) -> f64 {
    let height_diff = (to as f64 - from as f64).abs();
    let slope = height_diff / distance;

    let weight = 70.0; // Average weight in kg
    let altitude = (from as f64 + to as f64) / 2.0; // Approximate altitude
    let fatigue_factor = 0.1; // Example fatigue scaling
    let temperature = 20.0; // Moderate temperature in Celsius

    distance + slope_penalty(slope, weight, altitude, fatigue_factor, temperature)
}

/// Applies a penalty based on slope steepness and real-world effort factors.
fn slope_penalty(slope: f64, weight: f64, altitude: f64, fatigue_factor: f64, temperature: f64) -> f64 {
    let gravity_factor = 9.8 * weight;
    let altitude_penalty = if altitude > 2000.0 { 1.2 } else { 1.0 };
    let fatigue_penalty = 1.0 + fatigue_factor.min(0.5);
    let temp_penalty = if temperature < 0.0 || temperature > 35.0 { 1.1 } else { 1.0 };

    slope * gravity_factor * altitude_penalty * fatigue_penalty * temp_penalty
}

/// Generates valid neighbors of a given position.
fn neighbors(pos: (u32, u32), width: u32, height: u32) -> impl Iterator<Item = (u32, u32)> {
    let (x, y) = pos;
    [
        (x.wrapping_sub(1), y), // Left
        (x + 1, y),             // Right
        (x, y.wrapping_sub(1)), // Up
        (x, y + 1),             // Down
    ]
    .into_iter()
    .filter(move |&(nx, ny)| nx < width && ny < height)
}

/// Finds the optimal path between two points on a heightmap using A* algorithm.
fn find_optimal_path(
    heightmap: &GrayImage,
    start: (u32, u32),
    goal: (u32, u32),
) -> Option<(f64, Vec<(u32, u32)>)> {
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

    while let Some(Node { cost, priority, position }) = heap.pop() {
        if position == goal {
            let mut path = Vec::new();
            let mut current = position;
            while let Some(p) = prev[current.0 as usize][current.1 as usize] {
                path.push(current);
                current = p;
            }
            path.push(start);
            path.reverse();
            return Some((cost, path));
        }

        if visited.contains(&position) {
            continue;
        }

        visited.insert(position);

        for neighbor in neighbors(position, width, height) {
            let (nx, ny) = neighbor;
            let distance = if nx != position.0 && ny != position.1 { 1.414 } else { 1.0 };
            let new_cost = cost
                + effort(
                    heightmap.get_pixel(position.0, position.1)[0],
                    heightmap.get_pixel(nx, ny)[0],
                    distance,
                );

            if new_cost < dist[nx as usize][ny as usize] {
                dist[nx as usize][ny as usize] = new_cost;
                prev[nx as usize][ny as usize] = Some(position);
                heap.push(Node {
                    cost: new_cost,
                    priority: new_cost + heuristic((nx, ny), goal),
                    position: (nx, ny),
                });
            }
        }
    }

    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get command-line arguments for dynamic parameters
    let args: Vec<String> = env::args().collect();
    let weight: f64 = args.get(1).unwrap_or(&"70.0".to_string()).parse().unwrap_or(70.0);
    let fatigue_factor: f64 = args.get(2).unwrap_or(&"0.1".to_string()).parse().unwrap_or(0.1);
    let temperature: f64 = args.get(3).unwrap_or(&"20.0".to_string()).parse().unwrap_or(20.0);

    // Load heightmap as a dynamic image
    let img = ImageReader::open("heightmap.png")?.decode()?;

    // Validate that the image is grayscale
    validate_grayscale_image(&img)?;

    // Convert to grayscale image
    let gray_img = img.to_luma8();

    let start = (0, 0);
    let goal = (gray_img.width() - 1, gray_img.height() - 1);

    if let Some((cost, path)) = find_optimal_path(&gray_img, start, goal) {
        println!("Optimal path cost: {:.2}", cost);
        println!("Path: {:?}", path);
    } else {
        println!("No path found.");
    }

    Ok(())
}
