"""Shared GLSL shader code for audioglow visualizers."""

VERTEX_SHADER = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
"""

FRAG_COMMON_HEADER = """
#version 330
uniform sampler2D tex_bg;
uniform sampler2D tex_fg;
uniform float zoom;
uniform vec2 shake;
uniform float bloom;
uniform vec2 resolution;
uniform vec2 cover_res;
in vec2 v_uv;
out vec4 f_color;
"""

GLSL_GET_PALETTE_COLOR = """
uniform sampler2D tex_palette;
vec3 getPaletteColor(int index) {
    float u = (float(index) + 0.5) / 5.0;
    return texture(tex_palette, vec2(u, 0.5)).rgb;
}
"""

GLSL_GET_PARTICLE_COLOR = """
uniform sampler2D tex_overlay;
vec4 getParticleColor(vec4 color, vec2 uv) {
    vec3 target = vec3(0.321, 0.698, 0.008);
    float dist = distance(color.rgb, target);

    float threshold = 0.28;
    float softness  = 0.05;
    float key = smoothstep(threshold - softness, threshold + softness, dist);

    float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    float luma_mask = smoothstep(0.03, 0.10, luma);

    float pattern = (uv.y * 3.0 + uv.x * 2.0);
    int idx = clamp(int(mod(pattern, 5.0)), 0, 4);
    vec3 base_color = getPaletteColor(idx);

    float intensity = clamp(luma * 1.6, 0.0, 1.0);
    vec3 final_rgb = base_color * intensity;

    float alpha = key * luma_mask;
    return vec4(final_rgb, alpha);
}
"""

GLSL_SPECTRUM_FUNCTIONS = """
uniform sampler2D tex_spectrum;
float getVal(int index) {
    index = clamp(index, 0, 63);
    float u = (float(index) + 0.5) / 64.0;
    return texture(tex_spectrum, vec2(u, 0.5)).r;
}

float getBSplineAmp(float t) {
    float size = 64.0;
    float p = t * size;
    int i = int(floor(p));
    float f = fract(p);
    float p0 = getVal(i - 1); float p1 = getVal(i);
    float p2 = getVal(i + 1); float p3 = getVal(i + 2);
    float b0 = (1.0 - f)*(1.0 - f)*(1.0 - f)/6.0;
    float b1 = (3.0*f*f*f - 6.0*f*f + 4.0)/6.0;
    float b2 = (-3.0*f*f*f + 3.0*f*f + 3.0*f + 1.0)/6.0;
    float b3 = f*f*f/6.0;
    return p0*b0 + p1*b1 + p2*b2 + p3*b3;
}

vec4 drawWave(vec2 uv) {
    float x_screen = uv.x;
    float x_sampled = 0.04 + x_screen * 0.74;
    float amplitude = getBSplineAmp(x_sampled);

    amplitude = pow(amplitude, 2.0);
    amplitude = max(0.0, amplitude - 0.05);
    amplitude *= 1.7;

    float fade_mask = 1.0 - smoothstep(0.85, 1.0, x_screen);
    amplitude *= fade_mask;

    float max_h = 0.17;
    float wave_h = max_h * (1.0 - exp(-amplitude / max_h));
    float y_rel = 1.0 - uv.y;

    float glow_size = 0.03; float edge_sharp = 0.002; float stroke_size = 0.003;
    float mask_fill = 1.0 - smoothstep(wave_h, wave_h + edge_sharp, y_rel);
    float glow_mask = 1.0 - smoothstep(wave_h, wave_h + glow_size, y_rel);
    float glow_only = clamp(glow_mask - mask_fill, 0.0, 1.0);
    float stroke_top = wave_h + stroke_size;
    float mask_stroke = 1.0 - smoothstep(stroke_top, stroke_top + edge_sharp, y_rel);
    float outline = clamp(mask_stroke - mask_fill, 0.0, 1.0);

    int c_idx = clamp(int(x_screen * 4.0), 0, 4);
    vec3 c1 = getPaletteColor(c_idx);
    vec3 c2 = getPaletteColor(min(c_idx + 1, 4));
    vec3 col_fill = mix(c1, c2, fract(x_screen * 4.0));
    vec3 col_stroke = mix(col_fill, vec3(1.0), 0.35);

    vec4 res = vec4(0.0);
    res.rgb += col_fill * mask_fill;
    res.a = max(res.a, mask_fill * 0.95);
    res.rgb = mix(res.rgb, col_stroke, outline);
    res.a = max(res.a, outline);
    res.rgb += col_fill * glow_only * 0.6;
    res.a = max(res.a, glow_only * 0.5);
    return res;
}
"""

GLSL_COMPOSITE_COVER = """
vec4 compositeCover(vec2 uv_in) {
    vec2 center = vec2(0.5);
    vec2 uv = uv_in - shake;
    uv = (uv - center) / zoom + center;

    vec2 size_ratio = cover_res / resolution;
    vec2 tex_uv = (uv - center) / size_ratio + center;

    vec4 fg = vec4(0.0);
    if (tex_uv.x > 0.0 && tex_uv.x < 1.0 && tex_uv.y > 0.0 && tex_uv.y < 1.0) {
        fg = texture(tex_fg, tex_uv);
        fg.rgb *= (1.0 + bloom * 0.5);
    }
    return fg;
}
"""


def build_fragment_shader(*parts):
    """Concatenate GLSL parts into a complete fragment shader."""
    return "\n".join(parts)
