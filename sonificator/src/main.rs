use sonificator::Sonificator;

fn main() {
    let mut sonificator = Sonificator::new(48_000);
    let frame = vec![96u8, 156, 224, 255].repeat(64 * 64);
    sonificator.push_frame(&frame, 64, 64);
    let audio = sonificator.render(480);
    println!(
        "sonificator: {} stereo PCM frames rendered at {} Hz ({:.1} ms); feed push_frame() from your video player",
        audio.len() / 2,
        sonificator.sample_rate(),
        audio.len() as f32 / sonificator.sample_rate() as f32 / 2.0 * 1000.0
    );
}
