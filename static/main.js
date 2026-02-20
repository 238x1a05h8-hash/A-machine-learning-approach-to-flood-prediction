/* Rising Waters – Main JS */

// ── Form loading state ────────────────────────────
const form = document.getElementById('predictForm');
if (form) {
  form.addEventListener('submit', function () {
    const btn    = document.getElementById('submitBtn');
    const text   = document.getElementById('btnText');
    const loader = document.getElementById('btnLoader');
    if (btn && text && loader) {
      text.textContent = 'Analysing…';
      loader.classList.remove('hidden');
      btn.disabled = true;
    }
  });
}

// ── Auto-dismiss flash messages ───────────────────
document.querySelectorAll('.flash').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity .5s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 500);
  }, 5000);
});

// ── Particle field on hero ────────────────────────
const field = document.getElementById('particles');
if (field) {
  for (let i = 0; i < 40; i++) {
    const dot = document.createElement('span');
    dot.style.cssText = `
      position:absolute;
      width:${Math.random() * 3 + 1}px;
      height:${Math.random() * 3 + 1}px;
      border-radius:50%;
      background:rgba(0,200,255,${Math.random() * 0.4 + 0.1});
      left:${Math.random() * 100}%;
      top:${Math.random() * 100}%;
      animation: twinkle ${Math.random() * 4 + 3}s ease-in-out infinite;
      animation-delay:${Math.random() * 3}s;
    `;
    field.appendChild(dot);
  }
  const style = document.createElement('style');
  style.textContent = `
    @keyframes twinkle {
      0%,100% { opacity:.2; transform:scale(1); }
      50% { opacity:.9; transform:scale(1.6); }
    }
  `;
  document.head.appendChild(style);
}

// ── Scroll-reveal for feature cards ──────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      entry.target.style.transition = `opacity .5s ${i * 0.1}s, transform .5s ${i * 0.1}s`;
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'none';
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('.feature-card, .about-block, .p-step').forEach((el, i) => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(20px)';
  observer.observe(el);
});

// ── Input range highlight ─────────────────────────
document.querySelectorAll('input[type=number]').forEach(inp => {
  inp.addEventListener('input', function () {
    const min = parseFloat(this.min), max = parseFloat(this.max), val = parseFloat(this.value);
    if (!isNaN(val)) {
      this.style.borderColor = (val >= min && val <= max)
        ? 'rgba(0,217,126,.5)'
        : 'rgba(255,59,85,.5)';
    }
  });
});
