use std::env;
use std::io::{self, Read};
use std::process::{Command, Stdio};

use hound::{SampleFormat, WavSpec, WavWriter};
use sonificator::Sonificator;

const DEFAULT_SAMPLE_RATE: u32 = 48_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().filter(|arg| arg != "\\").collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!(
            "usage: cargo run -p video-music-test -- <video.mp4> [output.wav]\n\
             default output: video-music-test.wav"
        );
        std::process::exit(2);
    }

    let input = &args[1];
    let output = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("video-music-test.wav");
    let (width, height, fps) = video_info(input)?;
    let frame_size = width as usize * height as usize * 4;
    let frames_per_audio_block = DEFAULT_SAMPLE_RATE as f64 / fps;
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", input, "-f", "rawvideo", "-pix_fmt", "rgba", "-",
        ])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut video = ffmpeg
        .stdout
        .take()
        .ok_or("ffmpeg did not provide a video pipe")?;
    let spec = WavSpec {
        channels: 2,
        sample_rate: DEFAULT_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(output, spec)?;
    let mut sonificator = Sonificator::new(DEFAULT_SAMPLE_RATE);
    let mut frame = vec![0u8; frame_size];
    let mut total_frames = 0u64;
    let mut audio_position = 0.0f64;

    loop {
        match video.read_exact(&mut frame) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
        sonificator.push_frame(&frame, width, height);
        let next_audio_position = audio_position + frames_per_audio_block;
        let audio_frames = next_audio_position.floor() as usize - audio_position.floor() as usize;
        for sample in sonificator.render(audio_frames) {
            writer.write_sample((sample * i16::MAX as f32) as i16)?;
        }
        audio_position = next_audio_position;
        total_frames += 1;
        if total_frames % 30 == 0 {
            eprintln!("processed {:.1}s", total_frames as f64 / fps);
        }
    }

    writer.finalize()?;
    let status = ffmpeg.wait()?;
    if !status.success() {
        return Err(format!("ffmpeg exited with {status}").into());
    }
    println!(
        "wrote {output} ({:.1}s, {width}x{height} at {fps:.2} fps)",
        total_frames as f64 / fps
    );
    Ok(())
}

fn video_info(path: &str) -> Result<(u32, u32, f64), Box<dyn std::error::Error>> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0:s=,",
            path,
        ])
        .output()?;
    if !output.status.success() {
        return Err(format!("ffprobe exited with {}", output.status).into());
    }
    let line = String::from_utf8(output.stdout)?;
    let mut fields = line.trim().split(',');
    let width: u32 = fields.next().ok_or("missing video width")?.parse()?;
    let height: u32 = fields.next().ok_or("missing video height")?.parse()?;
    let rate = fields.next().ok_or("missing video frame rate")?;
    let (numerator, denominator) = rate.split_once('/').ok_or("invalid video frame rate")?;
    let fps = numerator.parse::<f64>()? / denominator.parse::<f64>()?;
    if width == 0 || height == 0 || fps <= 0.0 {
        return Err("video has invalid dimensions or frame rate".into());
    }
    Ok((width, height, fps))
}
