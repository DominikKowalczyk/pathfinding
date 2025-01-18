use criterion::{criterion_group, criterion_main, Criterion};
use super::*;

fn benchmark_find_optimal_path(c: &mut Criterion) {
    let img = ImageReader::open("heightmap.png").unwrap().decode().unwrap();
    let gray_img = img.to_luma8();
    let start = (0, 0);
    let goal = (gray_img.width() - 1, gray_img.height() - 1);
    let weight = 70.0;
    let fatigue_factor = 0.1;
    let temperature = 20.0;

    c.bench_function("find_optimal_path", |b| {
        b.iter(|| {
            find_optimal_path(&gray_img, start, goal, weight, fatigue_factor, temperature)
        })
    });
}

criterion_group!(benches, benchmark_find_optimal_path);
criterion_main!(benches);