"""Headless OpenGL renderer (moderngl + EGL / Mesa llvmpipe).

Draws: procedural sky (gradient, square sun & moon, stars), voxel meshes with
Minecraft-style face shading + ambient occlusion + sun light + distance fog,
emissive/flash tinting, alpha-blended and additive billboard particles.
"""
import numpy as np
import moderngl
from PIL import Image, ImageFilter
from . import textures as tx
from .geometry import perspective, look_at, VERT
from . import style

MAIN_VS = """
#version 330
uniform mat4 vp;
in vec3 in_pos; in vec2 in_uv; in vec3 in_norm;
in float in_ao; in float in_flash; in float in_alpha; in float in_emis;
out vec2 uv; out vec3 norm; out float ao; out float flash; out float alpha; out float emis; out vec3 wpos;
void main(){
    wpos = in_pos; uv = in_uv; norm = in_norm; ao = in_ao; flash = in_flash; alpha = in_alpha; emis = in_emis;
    gl_Position = vp * vec4(in_pos, 1.0);
}
"""
MAIN_FS = """
#version 330
uniform sampler2D atlas;
uniform vec3 cam; uniform vec3 sun_dir; uniform vec3 sun_col; uniform vec3 amb_col;
uniform vec3 fog_col; uniform float fog_start; uniform float fog_end;
uniform vec3 flash_col; uniform float exposure;
uniform float sun_power; uniform float rim_strength; uniform vec3 rim_col; uniform vec3 shadow_col;
in vec2 uv; in vec3 norm; in float ao; in float flash; in float alpha; in float emis; in vec3 wpos;
out vec4 frag;
void main(){
    vec4 t = texture(atlas, uv);
    if (t.a * alpha < 0.04) discard;
    vec3 n = normalize(norm);
    float face = 0.80 + 0.20 * n.y - 0.08 * abs(n.x);
    float diff = max(dot(n, sun_dir), 0.0);
    vec3 amb = mix(amb_col, amb_col * shadow_col, 1.0 - diff);
    vec3 light = amb * face + sun_col * diff * sun_power;
    vec3 col = t.rgb * light * ao * exposure;
    if (rim_strength > 0.0) {
        vec3 v = normalize(cam - wpos);
        float fres = pow(1.0 - max(dot(n, v), 0.0), 2.2);
        float back = clamp(dot(-v, sun_dir) * 0.6 + 0.4, 0.0, 1.0);
        float lit = max(dot(n, sun_dir), 0.0) * 0.6 + 0.4;
        col += rim_col * fres * back * lit * rim_strength * ao;
    }
    col = mix(col, t.rgb * 1.25, emis);
    col = mix(col, flash_col, clamp(flash, 0.0, 1.0));
    float d = length(wpos - cam);
    float f = clamp((d - fog_start) / max(fog_end - fog_start, 1.0), 0.0, 1.0);
    col = mix(col, fog_col, f * f * (3.0 - 2.0 * f));
    frag = vec4(col, t.a * alpha);
}
"""
SKY_VS = """
#version 330
in vec2 in_pos; out vec2 ndc;
void main(){ ndc = in_pos; gl_Position = vec4(in_pos, 0.9999, 1.0); }
"""
SKY_FS = """
#version 330
uniform mat4 inv_vp; uniform vec3 zenith; uniform vec3 horizon; uniform vec3 below;
uniform vec3 sun_dir; uniform vec3 moon_dir; uniform float stars; uniform float sun_vis; uniform float moon_vis;
uniform float sun_size; uniform float halo; uniform vec3 halo_col; uniform float clouds;
uniform vec3 cloud_lit; uniform vec3 cloud_dark;
in vec2 ndc; out vec4 frag;
float hash(vec3 p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float square_disc(vec3 d, vec3 c, float size){
    vec3 up = abs(c.y) > 0.95 ? vec3(1,0,0) : vec3(0,1,0);
    vec3 a = normalize(cross(up, c)); vec3 b = cross(c, a);
    float k = dot(d, c); if (k <= 0.0) return 0.0;
    vec2 q = vec2(dot(d, a), dot(d, b)) / k;
    return step(max(abs(q.x), abs(q.y)), size);
}
void main(){
    vec4 p0 = inv_vp * vec4(ndc, -1.0, 1.0); vec4 p1 = inv_vp * vec4(ndc, 1.0, 1.0);
    vec3 d = normalize(p1.xyz / p1.w - p0.xyz / p0.w);
    float h = d.y;
    vec3 col = mix(horizon, zenith, smoothstep(-0.02, 0.55, h));
    col = mix(col, below, smoothstep(0.0, -0.35, h));
    float sd = max(dot(d, sun_dir), 0.0);
    col += vec3(1.0, 0.75, 0.45) * pow(sd, 18.0) * 0.45 * sun_vis;
    if (halo > 0.0) {
        col += halo_col * (pow(sd, 8.0) * 0.35 + pow(sd, 60.0) * 0.9) * halo * sun_vis;
        col += halo_col * smoothstep(0.18, 0.0, abs(h - 0.02)) * 0.22 * halo * sun_vis * pow(sd, 2.0);
    }
    if (clouds > 0.0 && h > 0.015) {
        vec2 q = d.xz / (h + 0.06) * 1.6;
        vec2 cell = floor(q * vec2(2.0, 4.0));
        float n = hash(vec3(cell, 7.0));
        float n2 = hash(vec3(floor(q * vec2(1.0, 2.0)), 3.0));
        float c = step(0.66, n * 0.6 + n2 * 0.5) * smoothstep(0.015, 0.09, h) * smoothstep(0.55, 0.2, h);
        vec3 cc = mix(cloud_dark, cloud_lit, clamp(pow(sd, 3.0) * 1.4 + 0.15, 0.0, 1.0));
        col = mix(col, cc, c * clouds);
    }
    col = mix(col, vec3(1.0, 0.97, 0.75), square_disc(d, sun_dir, sun_size) * sun_vis);
    float md = square_disc(d, moon_dir, 0.05);
    col = mix(col, vec3(0.88, 0.9, 1.0), md * moon_vis);
    col += vec3(0.6, 0.65, 1.0) * pow(max(dot(d, moon_dir), 0.0), 40.0) * 0.25 * moon_vis;
    if (stars > 0.0 && h > 0.0) {
        vec3 g = floor(d * 260.0);
        float s = hash(g);
        if (s > 0.9965) col += vec3(1.0) * stars * (0.5 + 0.5 * hash(g + 1.0)) * smoothstep(0.0, 0.2, h);
    }
    frag = vec4(col, 1.0);
}
"""
PART_VS = """
#version 330
uniform mat4 vp; uniform vec3 cam_right; uniform vec3 cam_up;
in vec2 in_corner; in vec3 i_pos; in float i_size; in vec4 i_col;
out vec4 col; out vec3 wpos;
void main(){
    vec3 p = i_pos + (cam_right * in_corner.x + cam_up * in_corner.y) * i_size;
    wpos = p; col = i_col;
    gl_Position = vp * vec4(p, 1.0);
}
"""
PART_FS = """
#version 330
uniform vec3 cam; uniform vec3 fog_col; uniform float fog_start; uniform float fog_end;
in vec4 col; in vec3 wpos; out vec4 frag;
void main(){
    float d = length(wpos - cam);
    float f = clamp((d - fog_start) / max(fog_end - fog_start, 1.0), 0.0, 1.0);
    frag = vec4(mix(col.rgb, fog_col, f * 0.8), col.a * (1.0 - f * 0.5));
}
"""


class Renderer:
    def __init__(self, width, height, samples=4):
        self.w, self.h = width, height
        self.ctx = moderngl.create_standalone_context(backend="egl")
        ctx = self.ctx
        atlas = tx.get_atlas()
        self.atlas_img = atlas
        self.tex = ctx.texture((tx.ATLAS_PX, tx.ATLAS_PX), 4, atlas.tobytes())
        self.tex.build_mipmaps(0, 3)
        self.tex.filter = (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
        self.tex.anisotropy = 1.0

        self.prog = ctx.program(vertex_shader=MAIN_VS, fragment_shader=MAIN_FS)
        self.sky = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.part = ctx.program(vertex_shader=PART_VS, fragment_shader=PART_FS)

        tri = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        self.sky_vao = ctx.vertex_array(self.sky, [(ctx.buffer(tri.tobytes()), "2f", "in_pos")])
        corners = np.array([-.5, -.5, .5, -.5, .5, .5, -.5, -.5, .5, .5, -.5, .5], np.float32)
        self.corner_buf = ctx.buffer(corners.tobytes())

        self.fbo_ms = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((width, height), 4, samples=samples)],
            depth_attachment=ctx.depth_renderbuffer((width, height), samples=samples))
        self.fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((width, height), 4)])
        self.static = {}

    # ---------------------------------------------------------- meshes
    def _vao(self, verts):
        buf = self.ctx.buffer(np.ascontiguousarray(verts, np.float32).tobytes())
        vao = self.ctx.vertex_array(self.prog, [(buf, "3f 2f 3f 1f 1f 1f 1f",
                                                 "in_pos", "in_uv", "in_norm", "in_ao",
                                                 "in_flash", "in_alpha", "in_emis")])
        return buf, vao

    def set_static(self, key, verts):
        """Register / replace a cached mesh (e.g. terrain, island at a given build step)."""
        if key in self.static:
            return
        if len(verts) == 0:
            self.static[key] = None
            return
        self.static[key] = self._vao(verts)

    def drop_static(self, prefix):
        for k in [k for k in self.static if str(k).startswith(prefix)]:
            v = self.static.pop(k)
            if v:
                v[0].release(); v[1].release()

    def _draw_verts(self, verts):
        if verts is None or len(verts) == 0:
            return
        buf, vao = self._vao(verts)
        vao.render(moderngl.TRIANGLES)
        vao.release(); buf.release()

    def _draw_particles(self, parts, additive):
        if parts is None or len(parts) == 0:
            return
        ctx = self.ctx
        data = np.ascontiguousarray(parts, np.float32)          # N x 8: pos3 size1 rgba4
        ibuf = ctx.buffer(data.tobytes())
        vao = ctx.vertex_array(self.part, [(self.corner_buf, "2f", "in_corner"),
                                           (ibuf, "3f 1f 4f/i", "i_pos", "i_size", "i_col")])
        if additive:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        vao.render(moderngl.TRIANGLES, instances=len(data))
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        vao.release(); ibuf.release()

    # ---------------------------------------------------------- frame
    def render(self, fs):
        """fs: FrameSpec-like dict. Returns HxWx3 uint8."""
        ctx = self.ctx
        self.frame_no = getattr(self, "frame_no", -1) + 1
        if style.ACTIVE:
            fs = style.apply(fs, self.frame_no / 30.0)
        cam = fs["camera"]
        eye = np.array(cam["eye"], np.float32)
        view = look_at(eye, cam["target"], roll=cam.get("roll", 0.0))
        proj = perspective(cam.get("fov", 60), self.w / self.h, cam.get("near", 0.05), cam.get("far", 900))
        vp = proj @ view
        sky = fs["sky"]

        self.fbo_ms.use()
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.clear(*sky["horizon"], 1.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # sky
        s = self.sky
        s["inv_vp"].write(np.linalg.inv(vp).T.astype(np.float32).tobytes())
        for k in ("zenith", "horizon", "below"):
            s[k].value = tuple(sky[k])
        sun = np.array(sky["sun_dir"], np.float32); sun /= np.linalg.norm(sun)
        s["sun_dir"].value = tuple(sun)
        s["moon_dir"].value = tuple(-sun)
        s["stars"].value = sky.get("stars", 0.0)
        s["sun_vis"].value = sky.get("sun_vis", 1.0)
        s["moon_vis"].value = sky.get("moon_vis", 0.0)
        s["sun_size"].value = sky.get("sun_size", 0.07)
        s["halo"].value = sky.get("halo", 0.0)
        s["halo_col"].value = tuple(sky.get("halo_col", (1.0, 0.6, 0.3)))
        s["clouds"].value = sky.get("clouds", 0.0)
        s["cloud_lit"].value = tuple(sky.get("cloud_lit", (1.0, 0.7, 0.45)))
        s["cloud_dark"].value = tuple(sky.get("cloud_dark", (0.45, 0.3, 0.6)))
        self.sky_vao.render(moderngl.TRIANGLES)

        # world
        ctx.enable(moderngl.DEPTH_TEST)
        p = self.prog
        p["vp"].write(vp.T.astype(np.float32).tobytes())
        p["cam"].value = tuple(eye)
        light = fs["light"]
        ld = np.array(light["dir"], np.float32); ld /= np.linalg.norm(ld)
        p["sun_dir"].value = tuple(ld)
        p["sun_col"].value = tuple(light["sun"])
        p["amb_col"].value = tuple(light["ambient"])
        p["exposure"].value = light.get("exposure", 1.0)
        p["sun_power"].value = light.get("sun_power", 0.55)
        p["rim_strength"].value = light.get("rim", 0.0)
        p["rim_col"].value = tuple(light.get("rim_col", (1.0, 0.7, 0.4)))
        p["shadow_col"].value = tuple(light.get("shadow_col", (1.0, 1.0, 1.0)))
        fog = fs["fog"]
        for prog in (p, self.part):
            prog["fog_col"].value = tuple(fog["color"])
            prog["fog_start"].value = fog["start"]
            prog["fog_end"].value = fog["end"]
        p["flash_col"].value = tuple(fs.get("flash_col", (1, 1, 1)))
        self.tex.use(0)

        ctx.depth_mask = True
        for key in fs.get("static", []):
            m = self.static.get(key)
            if m:
                m[1].render(moderngl.TRIANGLES)
        self._draw_verts(fs.get("dynamic"))

        # particles + translucent (no depth write)
        right = view[0, :3]; up = view[1, :3]
        self.part["vp"].write(vp.T.astype(np.float32).tobytes())
        self.part["cam"].value = tuple(eye)
        self.part["cam_right"].value = tuple(right)
        self.part["cam_up"].value = tuple(up)
        ctx.depth_mask = False
        self._draw_particles(fs.get("particles"), additive=False)
        for key in fs.get("static_trans", []):
            m = self.static.get(key)
            if m:
                m[1].render(moderngl.TRIANGLES)
        self._draw_verts(fs.get("dynamic_trans"))
        self._draw_particles(fs.get("particles_add"), additive=True)
        ctx.depth_mask = True

        ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        raw = self.fbo.read(components=3)
        img = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)[::-1]
        post = dict(fs.get("post", {}))
        if post.get("god_rays", 0) > 0 and sky.get("sun_vis", 1.0) > 0.05:
            sp = vp @ np.array([*(eye + sun * 500.0), 1.0], np.float32)
            if sp[3] > 0:
                post["_sun_px"] = ((sp[0] / sp[3] * 0.5 + 0.5) * self.w, (0.5 - sp[1] / sp[3] * 0.5) * self.h)
        return post_process(img, post)


def _god_rays(f, sun_px, strength):
    """Screen-space light shafts: radial blur of the bright parts toward the sun."""
    h, w = f.shape[:2]
    sw, sh = w // 4, h // 4
    small = np.asarray(Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).resize((sw, sh), Image.BILINEAR))
    small = small.astype(np.float32) / 255.0
    lum = small.max(axis=2, keepdims=True)
    cx, cy = sun_px[0] / 4.0, sun_px[1] / 4.0
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    near = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (0.35 * sw) ** 2))[..., None]
    src = small * np.clip((lum - 0.82) / 0.18, 0, 1) * near
    acc = np.zeros_like(src)
    n, decay, wsum = 24, 1.0, 0.0
    for i in range(n):
        k = 1.0 - i * 0.022
        sx = np.clip(cx + (xx - cx) * k, 0, sw - 1).astype(np.int32)
        sy = np.clip(cy + (yy - cy) * k, 0, sh - 1).astype(np.int32)
        acc += src[sy, sx] * decay
        wsum += decay
        decay *= 0.93
    acc /= wsum
    rays = np.asarray(Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(2)).resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
    return f + rays * strength


def post_process(img, post):
    """Bloom, colour grade, vignette, white flash. Operates on uint8 HxWx3.

    Cinematic extras (all optional): bloom_threshold, bloom_wide, split toning
    (shadow_tint / highlight_tint), god_rays (needs _sun_px, set by the renderer).
    """
    f = img.astype(np.float32) / 255.0
    bloom = post.get("bloom", 0.35)
    h, w = img.shape[:2]
    if bloom > 0:
        thr = post.get("bloom_threshold", 0.72)
        small = Image.fromarray(img).resize((w // 8, h // 8), Image.BILINEAR)
        sa = np.asarray(small).astype(np.float32) / 255.0
        lum = sa.max(axis=2, keepdims=True)
        sa = sa * np.clip((lum - thr) / (1.0 - thr), 0, 1)
        sb = Image.fromarray((sa * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
        sb = np.asarray(sb.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
        f = f + sb * bloom * 1.6
        wide = post.get("bloom_wide", 0.0)
        if wide > 0:
            sw = Image.fromarray((sa * 255).astype(np.uint8)).resize((w // 32, h // 32), Image.BILINEAR)
            sw = sw.filter(ImageFilter.GaussianBlur(4))
            sw = np.asarray(sw.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
            f = f + sw * wide * 2.2
    if post.get("god_rays", 0) > 0 and "_sun_px" in post:
        f = _god_rays(f, post["_sun_px"], post["god_rays"])
    # gentle contrast/saturation grade
    sat = post.get("saturation", 1.12)
    g = f.mean(axis=2, keepdims=True)
    f = g + (f - g) * sat
    c = post.get("contrast", 1.06)
    f = (f - 0.5) * c + 0.5
    st, ht = post.get("shadow_tint"), post.get("highlight_tint")
    if st is not None or ht is not None:
        lum = np.clip(f.mean(axis=2, keepdims=True), 0, 1)
        if st is not None:
            f = f * (1 + (np.array(st, np.float32) - 1) * (1 - lum) ** 2)
        if ht is not None:
            f = f * (1 + (np.array(ht, np.float32) - 1) * lum ** 2)
    tint = post.get("tint")
    if tint is not None:
        f = f * np.array(tint, np.float32)
    vig = post.get("vignette", 0.35)
    if vig > 0:
        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        r2 = xx * xx * 0.8 + yy * yy * 0.6
        f = f * (1 - vig * np.clip(r2 - 0.25, 0, 1))[..., None]
    wf = post.get("white_flash", 0.0)
    if wf > 0:
        f = f + (1 - f) * wf
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)
