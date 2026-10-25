# Plain static GitHub Pages site.
# Demo audio and attribution

The seven demos use five openly licensed source recordings from Wikimedia Commons.
The released WAV files are mono, 32 kHz conversions of at most the first ten
seconds of each source. The original recordings are linked below.

| Demo source | Author / origin | License | Source page |
| --- | --- | --- | --- |
| Cough | ezwa / PDSounds | Public domain | [Cough 2](https://commons.wikimedia.org/wiki/File:Cough_2.ogg) |
| Dog bark | Edo.pt2 | CC0 1.0 | [Ladrido perro](https://commons.wikimedia.org/wiki/File:Ladrido_perro.ogg) |
| Rain and thunder | User:Caesar | Public domain | [Rain and thunder](https://commons.wikimedia.org/wiki/File:Rain_and_thunder.ogg) |
| Applause | Amada44 | CC0 1.0 | [Applause after a concert](https://commons.wikimedia.org/wiki/File:Sound_Effects_-_Applause_after_a_concert.ogg) |
| Ticking clock | ezwa / PDSounds | Public domain | [Alarm clock ticking](https://commons.wikimedia.org/wiki/File:Alarm_clock_ticking.ogg) |

The two challenge examples mix a 0.66-second crop of the cough recording into
the rain recording beginning at 4.00 seconds. The inserted cough's RMS level is
set relative to the underlying rain in that interval at -6 dB or -12 dB. The
yellow interval is known from this construction. It is not an independent event
annotation or a model output. These examples are demonstrations, not a measured
benchmark or a claim about general detection performance.

`scripts/build_demo_assets.py` regenerates the WAV, spectrogram, frame-score,
and manifest files using the bundled frozen model. The score files are
class-major little-endian float32 arrays in the exact class order of
`docs/data/manifest.json`. They contain post-CAMF 20 ms frame scores, not
ground-truth labels.

The Chinese glyph subset in `fonts/dasm-demo-cjk.woff2` is derived from
[Noto Sans CJK SC](https://github.com/notofonts/noto-cjk), renamed for this
site, and distributed under the SIL Open Font License 1.1. The full license
text is in `fonts/OFL.txt`.

Icons use the locally bundled Lucide 0.468.0 script. Its ISC license is in
`vendor/LUCIDE-LICENSE.txt`.
