// The page's small behaviours: reveal on scroll, the teaser's four recordings, the feature-similarity viewer, the
// project rail, and copying the BibTeX entry. A classic script rather than a module, so that index.html also works
// when it is opened straight from disk.
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- reveal on scroll ----------
  const revealer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add('in');
      revealer.unobserve(e.target);
    }
  }, { rootMargin: '0px 0px -8% 0px' });
  $$('.reveal').forEach((el) => revealer.observe(el));

  // ---------- teaser: the four recordings, one after another ----------
  // Each recording plays once, then the next one fades in. A chapter button jumps to its recording, and a click on
  // the video pauses or resumes. Nothing plays while the teaser is out of view.
  const stage = $('#stage');
  const chapters = $$('#chapters button');
  if (stage && chapters.length) {
    const vids = $$('video', stage);
    let cur = 0, visible = true, paused = reduced;
    for (const v of vids) { v.loop = false; v.autoplay = false; }

    const run = () => {
      const v = vids[cur];
      if (visible && !paused) v.play().catch(() => {}); else v.pause();
      stage.classList.toggle('is-paused', paused);
    };
    const show = (i) => {
      cur = (i + vids.length) % vids.length;
      vids.forEach((v, k) => {
        v.classList.toggle('on', k === cur);
        if (k !== cur) v.pause();
      });
      chapters.forEach((b, k) => b.classList.toggle('on', k === cur));
      vids[cur].currentTime = 0;
      vids[(cur + 1) % vids.length].preload = 'auto';
      run();
    };
    vids.forEach((v, k) => v.addEventListener('ended', () => { if (k === cur) show(cur + 1); }));
    chapters.forEach((b, k) => b.addEventListener('click', () => { paused = false; show(k); }));
    stage.addEventListener('click', () => { paused = !paused; run(); });
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; run(); }, { threshold: 0.35 }).observe(stage);

    const tick = () => {
      const v = vids[cur];
      chapters.forEach((b, k) => {
        $('.fill', b).style.width = k === cur && v.duration ? `${(v.currentTime / v.duration) * 100}%` : '0';
      });
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    show(0);
  }

  // ---------- segmented controls: the sliding pill follows the selected button ----------
  function placePill(seg) {
    const on = $('button[aria-selected="true"]', seg), pill = $('.pill', seg);
    if (!on || !pill) return;
    pill.style.width = `${on.offsetWidth}px`;
    pill.style.height = `${on.offsetHeight}px`;
    pill.style.transform = `translate(${on.offsetLeft}px, ${on.offsetTop - 4}px)`;
  }
  const segs = $$('.seg');
  segs.forEach(placePill);
  addEventListener('resize', () => segs.forEach(placePill));
  if (document.fonts) document.fonts.ready.then(() => segs.forEach(placePill));

  // ---------- behind the curtain: feature similarity, one view and example at a time ----------
  // The images are only loaded when they are shown. While the viewer is in sight, it steps through the examples and
  // views on its own. The active dot fills up until the next step, which is what times the steps: its animation
  // pauses while the pointer is on the image, while the viewer is out of sight, or after "Pause". A picked view or
  // example is shown a little longer before the viewer moves on.
  const heat = $('#heatView');
  const tabs = $('#heatTabs');
  if (heat && tabs) {
    const sets = $$('button', tabs);
    const dots = $('#heatDots');
    const foot = $('.heat-foot');
    const toggle = $('#heatAuto');
    const [imgA, imgB] = $$('.frame img', heat);
    const STEP_S = 4, PICKED_S = 9;
    let set = 0, ex = 0, front = imgA, back = imgB, visible = false, hover = false, paused = false, dur = STEP_S;

    const src = (s, i) => `static/images/heatmap_${sets[s].dataset.set}_${i + 1}.png`;
    const alt = (s) => `Left, a ${sets[s].dataset.view} radiograph with one position marked in red. Right, a second `
      + `${sets[s].dataset.view} radiograph overlaid with the similarity of every position to the marked one.`;

    function drawDots() {
      const n = +sets[set].dataset.n;
      dots.replaceChildren(...Array.from({ length: n }, (_, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.setAttribute('aria-label', `Example ${i + 1} of ${n}`);
        b.setAttribute('aria-current', String(i === ex));
        b.addEventListener('click', () => pick(set, i));
        if (i === ex) b.style.setProperty('--dur', `${dur}s`);
        return b;
      }));
    }

    function show(s, i) {
      const newSet = s !== set;
      set = s; ex = i;
      sets.forEach((b, k) => b.setAttribute('aria-selected', String(k === set)));
      placePill(tabs);
      drawDots();
      const url = src(set, ex);
      if (front.getAttribute('src') === url) return;
      back.onload = () => {
        back.classList.add('on'); front.classList.remove('on');
        back.alt = alt(set); back.removeAttribute('aria-hidden');
        front.alt = ''; front.setAttribute('aria-hidden', 'true');
        [front, back] = [back, front];
        // warm the cache for the next step
        const nx = next();
        new Image().src = src(nx[0], nx[1]);
      };
      back.src = url;
      if (newSet) heat.setAttribute('aria-label', sets[set].textContent);
    }

    const next = () => (ex + 1 < +sets[set].dataset.n ? [set, ex + 1] : [(set + 1) % sets.length, 0]);
    const pick = (s, i) => { dur = PICKED_S; show(s, i); };
    const sync = () => {
      foot.classList.toggle('is-paused', paused || hover || !visible);
      toggle.setAttribute('aria-pressed', String(paused));
      $('span', toggle).textContent = paused ? 'Play' : 'Pause';
    };

    // the active dot's fill ending is the step
    dots.addEventListener('animationend', () => { dur = STEP_S; show(...next()); });
    sets.forEach((b, k) => b.addEventListener('click', () => pick(k, 0)));
    toggle.addEventListener('click', () => { paused = !paused; sync(); });
    heat.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') { hover = true; sync(); } });
    heat.addEventListener('pointerleave', () => { hover = false; sync(); });
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync(); }, { threshold: 0.5 }).observe(heat);
    sync();
    drawDots();
  }

  // ---------- how it works: the animated schematic ----------
  // The bones and landmarks are those of the paper's lateral-knee measurement definitions, in their millimetre
  // units. The targets are that knee redrawn as other patients: another flexion, patellar height, bone shape,
  // exposure and pose, one of them with a plate. Every target point is the same deformation of its reference point,
  // so each correspondence, landmark and construction drawn is exact for its target: the figure shows the principle
  // and no result. The reference is annotated once (INTRO), then every cycle matches one target and turns it into
  // the next one. Where the figure is narrow, it is laid out top to bottom.
  const hiw = $('#hiw');
  if (hiw) {
    const NS = 'http://www.w3.org/2000/svg';
    const make = (tag, attrs, parent) => {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      if (parent) parent.append(e);
      return e;
    };
    const svg = $('svg', hiw);
    const { landmarks, grid } = JSON.parse($('#hiwData').textContent);
    const ISR = [0, 1, 3];   // superior pole, inferior pole, tibial tuberosity: d2 = patella, d1 = patellar tendon
    const BONES = ['fibula', 'tibia', 'femur', 'patella'];
    const outline = Object.fromEntries(BONES.map((b) => [b, $(`#hiw-${b}`).getAttribute('d').match(/-?[\d.]+/g).map(Number)]));

    // ---- the targets ----
    const KNEE = [50, 80], JOINT = [60, 66], PATELLA = [10, 54];
    const VARIANTS = [
      // extended, lower patella, thicker tibia with a plate, darker exposure
      { rot: 5, s: [1.04, 0.98], t: [-4, -2], flex: 6, pt: [-1, 5], prot: -3, bone: 0.3, tissue: 14, plate: 1,
        bulges: [{ c: [62, 112], k: [0.16, 0.02], s: 24, on: ['tibia', 'fibula'] }] },
      // flexed, higher and longer patella, larger condyles, brighter exposure
      { rot: -7, s: [0.95, 0.95], t: [3, -2], flex: -9, pt: [-1.5, -8], prot: 4, bone: 0.56, tissue: 34, plate: 0,
        bulges: [{ c: [52, 46], k: [0.16, 0.16], s: 16, on: ['femur'] }, { c: [10, 54], k: [0, 0.16], s: 14, on: ['patella'] }] },
    ];
    // a plate with four screws on the tibial shaft
    const PLATE = [[[60.5, 93], [65.5, 93], [65.5, 152], [60.5, 152]],
      ...[101, 115, 129, 143].map((y) => [[48, y - 0.9], [78, y - 0.9], [78, y + 0.9], [48, y + 0.9]])];
    const turn = (p, c, deg) => {
      const a = (deg * Math.PI) / 180, x = p[0] - c[0], y = p[1] - c[1];
      return [c[0] + x * Math.cos(a) - y * Math.sin(a), c[1] + x * Math.sin(a) + y * Math.cos(a)];
    };
    function warp(v, p, bone) {   // shape, then flexion and patellar height, then the pose of the whole image
      let q = [p[0], p[1]];
      for (const b of v.bulges) {
        if (!b.on.includes(bone)) continue;
        const dx = q[0] - b.c[0], dy = q[1] - b.c[1], g = Math.exp(-(dx * dx + dy * dy) / (2 * b.s * b.s));
        q = [q[0] + dx * b.k[0] * g, q[1] + dy * b.k[1] * g];
      }
      if (bone === 'femur') q = turn(q, JOINT, v.flex);
      if (bone === 'patella') q = turn(turn(q, PATELLA, v.prot).map((x, i) => x + v.pt[i]), JOINT, v.flex);   // moves with the femur
      q = [KNEE[0] + (q[0] - KNEE[0]) * v.s[0], KNEE[1] + (q[1] - KNEE[1]) * v.s[1]];
      q = turn(q, KNEE, v.rot);
      return [q[0] + v.t[0], q[1] + v.t[1]];
    }

    // only grid points that stay in view on every target (the dots are not clipped to the panels)
    const CROP = { x: -4, y: 16, w: 104, h: 138 };
    const inView = (q) => q[0] > CROP.x + 2 && q[0] < CROP.x + CROP.w - 2 && q[1] > CROP.y + 2 && q[1] < CROP.y + CROP.h - 2;
    const kept = grid.filter((g) => VARIANTS.every((v) => inView(warp(v, g, g[2]))));
    const cache = VARIANTS.map((v) => ({
      bones: Object.fromEntries(BONES.map((b) => {
        const f = outline[b], o = new Float64Array(f.length);
        for (let i = 0; i < f.length; i += 2) { const q = warp(v, [f[i], f[i + 1]], b); o[i] = q[0]; o[i + 1] = q[1]; }
        return [b, o];
      })),
      plate: PLATE.map((poly) => poly.map((p) => warp(v, p, 'tibia'))),
      dots: kept.map((g) => warp(v, g, g[2])),
      marks: landmarks.map((l) => warp(v, l.p, l.bone)),
    }));

    // ---- layout: left to right, or top to bottom; the markup holds the first ----
    const W = 300, H = 398;   // panel
    const LAYOUTS = {
      wide: { box: [960, 452], ref: [16, 16], tgt: [616, 16], enc: [466, 215], status: [466, 50], jitter: [70, 60],
        wires: ['M316 215 H378', 'M616 215 H554'], via: (j) => [466 + (j - 2.5) * 14, 106 + j * 6] },
      tall: { box: [420, 1080], ref: [50, 50], tgt: [50, 640], enc: [200, 563], status: [200, 22], jitter: [150, 36],
        wires: ['M200 486 V520', 'M200 640 V606'], via: (j) => [386 - j * 3, 563 + (j - 2.5) * 20] },
    };
    const ENC0 = [466, 215], STATUS0 = [466, 50];
    const panel = (px, py) => {
      const s = Math.min(W / CROP.w, H / CROP.h);
      return new DOMMatrix([s, 0, 0, s, px + (W - CROP.w * s) / 2 - CROP.x * s, py + (H - CROP.h * s) / 2 - CROP.y * s]);
    };
    const at = (m, p) => [m.a * p[0] + m.c * p[1] + m.e, m.b * p[0] + m.d * p[1] + m.f];
    let L = LAYOUTS.wide, REF = panel(...L.ref), TGT = panel(...L.tgt);

    // ---- elements ----
    const refUses = $$('#hiwRef use'), tgtPanel = $('#hiwTgt'), tgtPaths = $$('#hiwTgt path[data-bone]');
    const plate = $('#hiwTgt .implant'), platePaths = $$('path', plate);
    const dotLayer = $('#hiwDots'), markLayer = $('#hiwMarks'), trailLayer = $('#hiwTrails');
    const ys = kept.map((g) => g[1]), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const rand = (i) => { const x = Math.sin(i * 12.9898) * 43758.5453; return x - Math.floor(x); };   // fixed jitter
    const dots = kept.map((g, i) => {
      const u = (g[1] - y0) / (y1 - y0);
      const fill = `hsl(${195 + 135 * u + 18 * Math.sin(g[0] / 14)} 88% ${60 + 8 * Math.cos(g[0] / 20)}%)`;
      return { p: g, order: Math.min(1, u * 0.85 + rand(i) * 0.15), jx: rand(i + 7) - 0.5, jy: rand(i + 3) - 0.5,
        home: make('circle', { r: 2.8, fill }, dotLayer), fly: make('circle', { r: 2.8, fill }, dotLayer) };
    });
    const marks = landmarks.map((l) => ({ p: l.p, home: make('circle', { r: 0 }, markLayer), fly: make('circle', { r: 0 }, markLayer),
      trail: make('path', {}, trailLayer) }));

    // the Insall-Salvati construction, on the reference and on the target
    function construction(layer) {
      const g = { d2: make('path', {}, layer), d1: make('path', {}, layer) };
      g.t2 = make('text', {}, layer); g.t2.innerHTML = 'd<tspan dy="4">2</tspan>';
      g.t1 = make('text', {}, layer); g.t1.innerHTML = 'd<tspan dy="4">1</tspan>';
      return g;
    }
    function drawConstruction(g, P, u, label) {   // P: the three points in figure coordinates, u: 0..1 drawn
      const seg = (path, a, b, k) => {
        path.setAttribute('d', `M${a[0]} ${a[1]} L${b[0]} ${b[1]}`);
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        path.style.strokeDasharray = len;
        path.style.strokeDashoffset = len * (1 - Math.min(1, Math.max(0, k)));
      };
      seg(g.d2, P[0], P[1], u * 2);
      seg(g.d1, P[1], P[2], u * 2 - 1);
      const tag = (t, a, b, side, f) => {   // the label beside its line, a fraction f along it, as in the paper's figure
        const dx = b[0] - a[0], dy = b[1] - a[1], n = Math.hypot(dx, dy);
        let nx = -dy / n, ny = dx / n;
        if (Math.sign(nx) !== side) { nx = -nx; ny = -ny; }
        t.setAttribute('x', a[0] + dx * f + nx * 14 - 5);
        t.setAttribute('y', a[1] + dy * f + ny * 14 + 5);
        t.style.opacity = label;
      };
      tag(g.t2, P[0], P[1], 1, 0.4);
      tag(g.t1, P[1], P[2], -1, 0.5);
    }
    const refCons = construction($('#hiwRefCons')), tgtCons = construction($('#hiwTgtCons'));

    const wires = [$('#hiwWireA'), $('#hiwWireB')];
    const pkts = wires.map(() => make('circle', { r: 5, class: 'pkt' }, $('#hiwEnc').parentNode));
    const scans = $$('.scan', hiw), enc = $('#hiwEnc'), status = $('#hiwStatus'), statusText = $('#hiwStatus text');
    const tgtCap = $('#hiwTgtCap'), tgtSub = $('#hiwTgtSub');
    const stages = $$('#hiwStages li');
    const SUB = tgtSub.textContent, FORMULA = 'Insall-Salvati ratio = d₁ / d₂';

    // ---- the target in between two variants: i and j mixed by u ----
    let state = { i: 0, j: 1, u: 0 }, drawnKey = '';
    const mixed = (list, k) => {
      const a = cache[state.i][list][k], b = cache[state.j][list][k];
      return at(TGT, [a[0] + (b[0] - a[0]) * state.u, a[1] + (b[1] - a[1]) * state.u]);
    };
    function renderTarget() {
      const key = `${state.i}${state.j}${state.u.toFixed(3)}${L.box}`;
      if (key === drawnKey) return;
      drawnKey = key;
      const A = cache[state.i], B = cache[state.j], u = state.u, d = {};
      const pt = (a, b) => at(TGT, [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]);
      for (const name of BONES) {
        const a = A.bones[name], b = B.bones[name];
        let s = '';
        for (let k = 0; k < a.length; k += 2) {
          const q = pt([a[k], a[k + 1]], [b[k], b[k + 1]]);
          s += `${k ? 'L' : 'M'}${q[0].toFixed(1)} ${q[1].toFixed(1)}`;
        }
        d[name] = `${s}Z`;
      }
      for (const p of tgtPaths) p.setAttribute('d', d[p.dataset.bone]);
      platePaths.forEach((p, k) => p.setAttribute('d', `M${A.plate[k].map((q, n) => pt(q, B.plate[k][n]).map((x) => x.toFixed(1)).join(' ')).join(' L')}Z`));
      const va = VARIANTS[state.i], vb = VARIANTS[state.j], mix = (key2) => va[key2] + (vb[key2] - va[key2]) * u;
      plate.style.opacity = mix('plate');
      tgtPanel.style.setProperty('--bone-a', mix('bone').toFixed(3));
      tgtPanel.style.setProperty('--tissue', mix('tissue').toFixed(1));
    }

    function applyLayout(name) {
      L = LAYOUTS[name];
      hiw.classList.toggle('tall', name === 'tall');
      svg.setAttribute('viewBox', `0 0 ${L.box[0]} ${L.box[1]}`);
      for (const [key, [x, y]] of [['Ref', L.ref], ['Tgt', L.tgt]]) {
        for (const r of [$(`#hiw${key} .film`), $(`#hiwClip${key} rect`)]) { r.setAttribute('x', x); r.setAttribute('y', y); }
        $(`#hiw${key} .scan`).setAttribute('x', x);
        const cap = $(`#hiw${key} .cap`);
        cap.setAttribute('x', x + W / 2); cap.setAttribute('y', y + H + 26);
      }
      $$('.ghost', hiw).forEach((g, i) => { g.setAttribute('x', L.tgt[0] + 32 - 16 * i); g.setAttribute('y', L.tgt[1] - 20 + 10 * i); });
      wires.forEach((w, i) => w.setAttribute('d', L.wires[i]));
      enc.setAttribute('transform', `translate(${L.enc[0] - ENC0[0]} ${L.enc[1] - ENC0[1]})`);
      status.setAttribute('transform', `translate(${L.status[0] - STATUS0[0]} ${L.status[1] - STATUS0[1]})`);
      REF = panel(...L.ref); TGT = panel(...L.tgt);
      for (const u of refUses) u.setAttribute('transform', REF.toString());
      for (const d of dots) { d.from = at(REF, d.p); d.via = [L.enc[0] + d.jx * L.jitter[0], L.enc[1] + d.jy * L.jitter[1]]; }
      marks.forEach((m, j) => { m.from = at(REF, m.p); m.via = L.via(j); });
      drawnKey = '';
    }
    const pickLayout = () => (hiw.clientWidth < 700 ? 'tall' : 'wide');
    let layout = pickLayout();
    applyLayout(layout);

    // ---- the timeline ----
    const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
    const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2);
    const pop = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : 1 + 2.2 * (u - 1) ** 3 + 1.2 * (u - 1) ** 2);
    const win = (t, a, b, d = 0.25) => clamp(Math.min((t - a) / d, (b - t) / d));
    const bez = (a, c, b, u) => [(1 - u) ** 2 * a[0] + 2 * (1 - u) * u * c[0] + u * u * b[0], (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * c[1] + u * u * b[1]];
    const put = (c, p, r, o) => { c.setAttribute('cx', p[0].toFixed(1)); c.setAttribute('cy', p[1].toFixed(1)); c.setAttribute('r', r); c.style.opacity = o; };

    const INTRO = 3.0, CYCLE = 11.4;
    const STATUS_INTRO = '1 · An expert annotates once';
    const STATUS = [[0, 1.3, '2 · Shared encoder reads both'], [1.3, 4.6, '2 · Every position is matched'],
      [4.6, 6.5, '3 · Landmarks follow the matches'], [6.5, 10.2, '3 · Measurements follow'],
      [10.2, CYCLE, 'Next radiograph of the cohort']];

    function setStatus(text, o) {
      if (statusText.textContent !== text) {
        statusText.textContent = text;
        const w = statusText.getComputedTextLength() + 36, r = $('rect', status);
        r.setAttribute('width', w); r.setAttribute('x', STATUS0[0] - w / 2);
      }
      status.style.opacity = o;
    }
    const setStage = (k) => stages.forEach((li, i) => li.classList.toggle('on', i === k));

    function draw(t) {
      // ---- the reference: annotated once, then kept ----
      const refPts = ISR.map((i) => marks[i].from);
      marks.forEach((m, j) => put(m.home, m.from, (4.2 * pop(clamp((t - 0.5 - j * 0.22) / 0.3))).toFixed(2), 1));
      drawConstruction(refCons, refPts, ease(clamp((t - 1.9) / 0.7)), clamp((t - 2.5) / 0.3));

      if (t < INTRO) {
        setStage(0);
        setStatus(STATUS_INTRO, win(t, 0.15, INTRO - 0.1, 0.25));
        state = { i: 0, j: 1, u: 0 };
        renderTarget();
        for (const d of dots) { d.home.style.opacity = 0; d.fly.style.opacity = 0; }
        for (const m of marks) { m.fly.style.opacity = 0; m.trail.style.opacity = 0; }
        drawConstruction(tgtCons, refPts, 0, 0);
        pkts.forEach((p) => { p.style.opacity = 0; });
        scans.forEach((s) => { s.style.opacity = 0; });
        return;
      }
      const k = Math.floor((t - INTRO) / CYCLE), c = (t - INTRO) % CYCLE, n = VARIANTS.length;
      state = { i: k % n, j: (k + 1) % n, u: ease(clamp((c - 10.3) / 0.9)) };
      renderTarget();
      tgtCap.textContent = `Target ${((state.u > 0.5 ? k + 1 : k) % n) + 1}`;
      tgtSub.textContent = c >= 7.2 && c < 10.3 ? FORMULA : SUB;
      setStage(c < 4.6 ? 1 : 2);
      const s = STATUS.find(([a, b]) => c >= a && c < b);
      setStatus(s[2], win(c, s[0], s[1], 0.2));

      // ---- 2: the shared encoder reads both images, then every position is matched ----
      const scan = clamp(c / 1.0);
      scans.forEach((r, i) => { r.setAttribute('y', (i ? L.tgt : L.ref)[1] + scan * (H - 30) - 40); r.style.opacity = win(c, 0, 1.05, 0.15); });
      pkts.forEach((p, i) => {
        const w = wires[i], q = w.getPointAtLength(w.getTotalLength() * ease(clamp((c - 0.15) / 0.8)));
        p.setAttribute('cx', q.x); p.setAttribute('cy', q.y);
        p.style.opacity = win(c, 0.15, 0.95, 0.1);
      });
      enc.classList.toggle('on', c >= 0.6 && c < 1.6);

      const fadeOut = 1 - clamp((c - 10.0) / 0.35);
      const dim = 1 - 0.7 * clamp((c - 4.6) / 0.4);
      dots.forEach((d, i) => {
        const show = clamp((c - 1.2 - d.order * 0.5) / 0.25);
        put(d.home, d.from, 2.8, show * (1 - 0.45 * clamp((c - 3.4) / 0.6)) * dim * fadeOut);
        const u = clamp((c - 1.6 - d.order * 1.3) / 1.0);
        if (u <= 0) { d.fly.style.opacity = 0; return; }
        const to = mixed('dots', i);
        put(d.fly, u < 1 ? bez(d.from, d.via, to, ease(u)) : to, 2.8, dim * fadeOut);
      });

      // ---- 3: the landmarks follow the matches, then the measurement is drawn ----
      marks.forEach((m, j) => {
        const a = 4.8 + j * 0.14, u = clamp((c - a) / 1.1), to = mixed('marks', j);
        if (u <= 0) { m.fly.style.opacity = 0; m.trail.style.opacity = 0; return; }
        const land = clamp((c - a - 1.1) / 0.3);
        put(m.fly, u < 1 ? bez(m.from, m.via, to, ease(u)) : to, (u < 1 ? 3.4 : 3.4 + 0.8 * pop(land)).toFixed(2), fadeOut);
        m.trail.setAttribute('d', `M${m.from[0]} ${m.from[1]} Q${m.via[0]} ${m.via[1]} ${to[0]} ${to[1]}`);
        const len = m.trail.getTotalLength();
        m.trail.style.strokeDasharray = len;
        m.trail.style.strokeDashoffset = len * (1 - ease(u));
        m.trail.style.opacity = 1 - clamp((c - 7.4) / 0.6);
      });
      drawConstruction(tgtCons, ISR.map((i) => mixed('marks', i)), ease(clamp((c - 6.5) / 0.7)) * fadeOut, clamp((c - 7.1) / 0.3) * fadeOut);
    }

    function still() {   // without motion: the reference and the first target, both finished
      hiw.classList.remove('playing');
      draw(INTRO + 9.0);
      status.style.opacity = 0;
      stages.forEach((li) => li.classList.remove('on'));
    }

    let t = 0;
    new ResizeObserver(() => {
      const next = pickLayout();
      if (next === layout) return;
      layout = next;
      applyLayout(layout);
      if (reduced) still(); else draw(t);
    }).observe(hiw);

    if (reduced) still();
    else {
      hiw.classList.add('playing');
      draw(0);
      let raf = 0, last = 0, visible = false;
      const frame = (now) => {
        t += Math.min(0.1, (now - last) / 1000);   // keeps its place while out of view or in a hidden tab
        last = now;
        draw(t);
        raf = requestAnimationFrame(frame);
      };
      const start = () => { if (!raf && visible && !document.hidden) { last = performance.now(); raf = requestAnimationFrame(frame); } };
      const stop = () => { cancelAnimationFrame(raf); raf = 0; };
      new IntersectionObserver(([e]) => { visible = e.isIntersecting; visible ? start() : stop(); }, { threshold: 0.3 }).observe(hiw);
      document.addEventListener('visibilitychange', () => (document.hidden ? stop() : start()));
    }
  }

  // ---------- accompanying and downstream projects ----------
  // A card grows out of the rail's common height while its full text or its BibTeX entry is shown.
  for (const card of $$('.pcard')) {
    const more = $('.more', card), bibBtn = $('.bib-toggle', card), bib = $('pre', card);
    const sync = () => card.classList.toggle('is-open', card.classList.contains('is-read') || (!!bib && !bib.hidden));
    if (more) more.addEventListener('click', () => {
      const read = card.classList.toggle('is-read');
      more.setAttribute('aria-expanded', String(read));
      $('span', more).textContent = read ? 'Read less' : 'Read more';
      sync();
    });
    if (bibBtn && bib) bibBtn.addEventListener('click', () => {
      bib.hidden = !bib.hidden;
      bibBtn.setAttribute('aria-expanded', String(!bib.hidden));
      sync();
    });
  }

  // the rail: drag with the mouse, scroll with a wheel or trackpad, swipe on touch, or step with the chevrons
  const rail = $('.rail');
  const track = rail && $('.rail-track', rail);
  if (track) {
    const prev = $('.rail-nav.prev', rail), nextBtn = $('.rail-nav.next', rail);
    const progress = $('#railProgress');
    const cards = $$('.pcard', track);
    const EDGE = 8;   // sub-pixel layout offsets, in px

    const stepSize = () => cards[0].getBoundingClientRect().width + (parseFloat(getComputedStyle(track).columnGap) || 20);
    const pages = () => Math.max(1, Math.round((track.scrollWidth - track.clientWidth) / stepSize()) + 1);

    function update() {
      const max = track.scrollWidth - track.clientWidth;
      const scrollable = max > EDGE;
      const atStart = track.scrollLeft <= EDGE, atEnd = track.scrollLeft >= max - EDGE;
      rail.classList.toggle('fade-start', scrollable && !atStart);
      rail.classList.toggle('fade-end', scrollable && !atEnd);
      prev.hidden = !scrollable || atStart;
      nextBtn.hidden = !scrollable || atEnd;
      const n = scrollable ? pages() : 0;
      if (progress.childElementCount !== n) progress.replaceChildren(...Array.from({ length: n }, () => document.createElement('i')));
      const at = atEnd ? n - 1 : Math.round(track.scrollLeft / stepSize());
      $$('i', progress).forEach((d, k) => d.classList.toggle('on', k === at));
    }
    track.addEventListener('scroll', update, { passive: true });
    addEventListener('resize', update);
    new ResizeObserver(update).observe(track);
    update();

    prev.addEventListener('click', () => track.scrollBy({ left: -stepSize(), behavior: 'smooth' }));
    nextBtn.addEventListener('click', () => track.scrollBy({ left: stepSize(), behavior: 'smooth' }));

    // drag to pan, with the mouse only: touch and pen keep their native scrolling
    const DRAG = 5;   // px before a press counts as a drag, not a click
    let id = null, x0 = 0, s0 = 0, dragging = false, justDragged = false;
    track.addEventListener('pointerdown', (e) => {
      if (e.pointerType !== 'mouse' || e.button !== 0) return;
      id = e.pointerId; x0 = e.clientX; s0 = track.scrollLeft; dragging = false;
    });
    track.addEventListener('pointermove', (e) => {
      if (id === null || e.pointerId !== id) return;
      const dx = e.clientX - x0;
      if (!dragging) {
        if (Math.abs(dx) < DRAG) return;
        dragging = true;
        track.classList.add('is-dragging');
        try { track.setPointerCapture(id); } catch { /* capture is a nicety */ }
      }
      track.scrollLeft = s0 - dx;
      e.preventDefault();
    });
    const end = (e) => {
      if (id === null || (e && e.pointerId !== id)) return;
      if (dragging) {
        track.classList.remove('is-dragging');
        try { track.releasePointerCapture(id); } catch { /* nothing to release */ }
        // the click that ends a drag must not trigger a card's button
        justDragged = true;
        setTimeout(() => { justDragged = false; }, 0);
      }
      id = null; dragging = false;
    };
    track.addEventListener('pointerup', end);
    track.addEventListener('pointercancel', end);
    track.addEventListener('lostpointercapture', end);
    track.addEventListener('click', (e) => { if (justDragged) { e.preventDefault(); e.stopPropagation(); } }, true);
    track.addEventListener('dragstart', (e) => e.preventDefault());
  }

  // ---------- copy the BibTeX entry ----------
  const copy = $('#copyBib');
  if (copy) {
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText($('#bib').innerText.trim());
        copy.textContent = 'Copied';
      } catch {
        copy.textContent = 'Select and copy';
      }
      setTimeout(() => { copy.textContent = 'Copy'; }, 1800);
    });
  }
})();
