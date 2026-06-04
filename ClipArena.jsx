import React, { useState, useMemo } from "react";
import { ChevronRight, Zap, Crosshair, Trophy, ArrowLeft, Check, AlertTriangle, Activity } from "lucide-react";

/* ============================================================
   Clip Arena — Lacrosse IQ System (in-memory MVP prototype)
   Implements: predict -> diagnose -> adjust loop, normalized
   entropy (disagreement), anti-leak gate, Brier scoring,
   reputation leaderboard. No backend; seeds synthetic priors.
   ============================================================ */

const PALETTE = {
  bg: "#0c0e0d",
  panel: "#15181a",
  panel2: "#1c2023",
  line: "#2b3034",
  ink: "#eef2ee",
  mute: "#7d8a82",
  acid: "#c8f032",   // primary accent (scoreboard lime)
  amber: "#f0a832",
  red: "#e6553f",
};

const DISPLAY = "'Bebas Neue', 'Arial Narrow', sans-serif";
const BODY = "'Spline Sans', system-ui, sans-serif";

// confidence(1..5) -> probability mass on chosen option (k=5 calibration)
const Q = [0.4, 0.55, 0.7, 0.85, 0.95];

const CAT_COLORS = {
  CLEAR: "#c8f032", RIDE: "#f0a832", EMO: "#5ad1c4",
  MAN_DOWN: "#e6553f", TRANSITION: "#9b8cf0", DEFENSE: "#6fa8ff",
};

const CLIPS = [
  {
    id: "c1", category: "CLEAR", title: "Failed clear vs. 10-man ride, 4Q",
    context: { score: "Down 1", time: "2:40 / 4Q", manpower: "Even" },
    predict: {
      text: "What happens on this possession?",
      options: ["Successful clear", "Turnover", "Shot against", "Timeout / reset", "Penalty"],
      field: [9, 31, 6, 8, 6], elite: [3, 14, 1, 4, 2],
    },
    diagnose: {
      text: "Primary factor driving the outcome?",
      options: ["Ride pressure", "Outlet structure", "Spacing", "Goalie decision", "Matchup"],
      field: [22, 19, 10, 5, 4],
    },
    adjust: {
      text: "Correct coaching adjustment?",
      options: ["Change clear structure", "Outlet to sideline", "Burn timeout", "Switch personnel", "Hold pattern"],
      field: [24, 14, 11, 8, 3],
    },
    outcome: { actual: 1, summary: "Pressured turnover at midline; sideline outlet was open but unused.", keyFailure: "Ball-side wing failed to release on the slide.", process: "DEFENSIBLE" },
  },
  {
    id: "c2", category: "EMO", title: "EMO set, 1:10 remaining, down 2",
    context: { score: "Down 2", time: "1:10 / 4Q", manpower: "EMO (6v5)" },
    predict: {
      text: "What happens on this EMO possession?",
      options: ["Goal", "Save", "Turnover", "Shot wide", "Reset for last shot"],
      field: [18, 12, 9, 7, 21], elite: [11, 4, 2, 2, 14],
    },
    diagnose: {
      text: "Primary factor here?",
      options: ["Ball movement tempo", "Off-ball spacing", "Pipe presence", "Slide read", "Shooter selection"],
      field: [12, 24, 8, 14, 9],
    },
    adjust: {
      text: "Correct adjustment?",
      options: ["Invert the set", "Skip-pass earlier", "Plant a pipe", "Slow tempo / reset", "Iso the matchup"],
      field: [10, 19, 17, 13, 8],
    },
    outcome: { actual: 0, summary: "Skip to weakside finisher after the X-slide committed; goal.", keyFailure: null, process: "SOUND" },
  },
  {
    id: "c3", category: "DEFENSE", title: "Unsettled defense, transition 4v3",
    context: { score: "Tied", time: "8:15 / 3Q", manpower: "4v3 against" },
    predict: {
      text: "Result of this defensive sequence?",
      options: ["Stop / clear", "Goal against", "Shot saved", "Penalty drawn", "Turnover forced"],
      field: [14, 17, 11, 5, 9], elite: [9, 5, 6, 3, 8],
    },
    diagnose: {
      text: "Primary factor?",
      options: ["Slide timing", "Hot/communication", "Recovery speed", "Crease coverage", "Top-side angle"],
      field: [21, 16, 8, 6, 5],
    },
    adjust: {
      text: "Correct adjustment?",
      options: ["Slide earlier", "Zone the 4v3", "Protect the pipe", "Take away inside", "Match speed back"],
      field: [13, 9, 18, 14, 4],
    },
    outcome: { actual: 4, summary: "Early slide + recovery forces a contested skip; thrown away.", keyFailure: null, process: "SOUND" },
  },
];

const SEED_BOARD = [
  { name: "K. Marsh", expert: true, rep: 71.4 },
  { name: "D. Okoro", expert: true, rep: 64.8 },
  { name: "R. Vance", expert: false, rep: 52.1 },
  { name: "T. Halloran", expert: false, rep: 40.6 },
  { name: "S. Pell", expert: false, rep: 28.3 },
];

/* ---------- math ---------- */
function normEntropy(counts) {
  const N = counts.reduce((a, b) => a + b, 0);
  const k = counts.length;
  if (N === 0 || k <= 1) return 0;
  let H = 0;
  for (const n of counts) { if (n > 0) { const p = n / N; H -= p * Math.log(p); } }
  return H / Math.log(k);
}
function topShare(counts) {
  const N = counts.reduce((a, b) => a + b, 0);
  return N === 0 ? 0 : Math.max(...counts) / N;
}
function tvDivergence(a, b) {
  const Na = a.reduce((x, y) => x + y, 0), Nb = b.reduce((x, y) => x + y, 0);
  if (!Na || !Nb) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] / Na - b[i] / Nb);
  return 0.5 * s;
}
function brier(choiceIdx, confidence, actualIdx, k) {
  const q = Q[confidence - 1];
  const rest = (1 - q) / (k - 1);
  let B = 0;
  for (let i = 0; i < k; i++) {
    const p = i === choiceIdx ? q : rest;
    const o = i === actualIdx ? 1 : 0;
    B += (p - o) * (p - o);
  }
  return B;
}
function brierChance(k) {
  const p = 1 / k;
  return (k - 1) * p * p + (1 - p) * (1 - p);
}

/* ---------- small UI atoms ---------- */
const Tag = ({ cat }) => (
  <span style={{ fontFamily: DISPLAY, letterSpacing: "0.08em", color: PALETTE.bg, background: CAT_COLORS[cat] || PALETTE.acid }}
    className="px-2 py-0.5 text-sm">{cat.replace("_", " ")}</span>
);

function DistBar({ options, counts, highlight, actual }) {
  const N = counts.reduce((a, b) => a + b, 0);
  return (
    <div className="space-y-1.5">
      {options.map((opt, i) => {
        const pct = N ? Math.round((counts[i] / N) * 100) : 0;
        const isPick = i === highlight;
        const isActual = i === actual;
        return (
          <div key={i} className="flex items-center gap-3">
            <div className="w-40 shrink-0 text-sm flex items-center gap-1.5" style={{ color: isActual ? PALETTE.acid : PALETTE.ink }}>
              {isActual && <Check size={13} style={{ color: PALETTE.acid }} />}{opt}
            </div>
            <div className="flex-1 h-5 relative" style={{ background: PALETTE.panel2 }}>
              <div className="h-full" style={{ width: `${pct}%`, background: isActual ? PALETTE.acid : isPick ? PALETTE.amber : PALETTE.line }} />
              <span className="absolute right-1.5 top-0 text-xs leading-5" style={{ color: PALETTE.mute }}>{pct}%</span>
            </div>
            {isPick && <span className="text-xs shrink-0" style={{ color: PALETTE.amber, fontFamily: DISPLAY, letterSpacing: "0.05em" }}>YOU</span>}
          </div>
        );
      })}
    </div>
  );
}

function Metric({ label, value, sub, color }) {
  return (
    <div style={{ borderColor: PALETTE.line }} className="border px-3 py-2 flex-1">
      <div className="text-xs uppercase tracking-wider" style={{ color: PALETTE.mute }}>{label}</div>
      <div style={{ fontFamily: DISPLAY, color: color || PALETTE.ink }} className="text-3xl leading-none mt-1">{value}</div>
      {sub && <div className="text-xs mt-0.5" style={{ color: PALETTE.mute }}>{sub}</div>}
    </div>
  );
}

/* ---------- decision loop ---------- */
function ClipView({ clip, prior, onSubmit, onBack }) {
  const [step, setStep] = useState(prior ? 3 : 0);
  const [predict, setPredict] = useState(prior?.predict ?? null);
  const [confidence, setConfidence] = useState(prior?.confidence ?? 3);
  const [diagnose, setDiagnose] = useState(prior?.diagnose ?? null);
  const [adjust, setAdjust] = useState(prior?.adjust ?? null);

  const k = clip.predict.options.length;
  const answered = step >= 3;

  // live distributions including the user's pick
  const dist = useMemo(() => {
    const add = (base, idx) => base.map((c, i) => c + (i === idx ? 1 : 0));
    return {
      predict: add(clip.predict.field, predict),
      diagnose: add(clip.diagnose.field, diagnose),
      adjust: add(clip.adjust.field, adjust),
    };
  }, [clip, predict, diagnose, adjust]);

  const submit = () => {
    const B = brier(predict, confidence, clip.outcome.actual, k);
    onSubmit({ predict, confidence, diagnose, adjust, brier: B });
    setStep(3);
  };

  const Step = ({ n, label, prompt, value, setValue }) => (
    <div style={{ borderColor: PALETTE.line, opacity: step === n || answered ? 1 : 0.35 }} className="border p-4">
      <div className="flex items-center gap-2 mb-2">
        <span style={{ fontFamily: DISPLAY, background: PALETTE.acid, color: PALETTE.bg }} className="px-2 text-sm">{n + 1}</span>
        <span className="text-xs uppercase tracking-widest" style={{ color: PALETTE.mute }}>{label}</span>
      </div>
      <div className="mb-3" style={{ color: PALETTE.ink }}>{prompt.text}</div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {prompt.options.map((opt, i) => {
          const sel = value === i;
          return (
            <button key={i} disabled={step !== n && !answered}
              onClick={() => { setValue(i); if (!answered && step === n && n < 2) setStep(n + 1); }}
              style={{ borderColor: sel ? PALETTE.acid : PALETTE.line, background: sel ? PALETTE.panel2 : "transparent", color: sel ? PALETTE.acid : PALETTE.ink, cursor: step !== n && !answered ? "default" : "pointer" }}
              className="border px-3 py-2 text-left text-sm flex items-center justify-between">
              <span>{opt}</span>{sel && <Check size={15} />}
            </button>
          );
        })}
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      <button onClick={onBack} className="flex items-center gap-1 text-sm" style={{ color: PALETTE.mute }}>
        <ArrowLeft size={14} /> Feed
      </button>

      <div className="flex items-center gap-3">
        <Tag cat={clip.category} />
        <h2 style={{ fontFamily: DISPLAY, color: PALETTE.ink }} className="text-2xl tracking-wide">{clip.title}</h2>
      </div>

      {/* video placeholder */}
      <div style={{ background: "#000", borderColor: PALETTE.line }} className="border aspect-video flex items-center justify-center relative">
        <Crosshair size={40} style={{ color: PALETTE.line }} />
        <div className="absolute bottom-2 left-2 flex gap-3 text-xs" style={{ color: PALETTE.mute }}>
          <span>{clip.context.score}</span><span>{clip.context.time}</span><span>{clip.context.manpower}</span>
        </div>
      </div>

      <Step n={0} label="Predict" prompt={clip.predict} value={predict} setValue={setPredict} />

      {(step >= 0 && predict !== null && !answered) && (
        <div style={{ borderColor: PALETTE.line }} className="border p-4">
          <div className="text-xs uppercase tracking-widest mb-2" style={{ color: PALETTE.mute }}>Confidence</div>
          <div className="flex gap-2">
            {[1, 2, 3, 4, 5].map((c) => (
              <button key={c} onClick={() => setConfidence(c)}
                style={{ borderColor: confidence === c ? PALETTE.acid : PALETTE.line, color: confidence === c ? PALETTE.acid : PALETTE.mute, fontFamily: DISPLAY }}
                className="border w-10 h-10 text-xl">{c}</button>
            ))}
          </div>
        </div>
      )}

      <Step n={1} label="Diagnose" prompt={clip.diagnose} value={diagnose} setValue={setDiagnose} />
      <Step n={2} label="Adjust" prompt={clip.adjust} value={adjust} setValue={setAdjust} />

      {!answered && (
        <button disabled={predict === null || diagnose === null || adjust === null} onClick={submit}
          style={{ background: predict !== null && diagnose !== null && adjust !== null ? PALETTE.acid : PALETTE.line, color: PALETTE.bg, fontFamily: DISPLAY }}
          className="w-full py-3 text-xl tracking-widest flex items-center justify-center gap-2">
          LOCK IN <ChevronRight size={20} />
        </button>
      )}

      {answered && <Reveal clip={clip} dist={dist} pick={prior ?? { predict, confidence, diagnose, adjust }} />}
    </div>
  );
}

/* ---------- reveal (gated behind submission) ---------- */
function Reveal({ clip, dist, pick }) {
  const k = clip.predict.options.length;
  const Hp = normEntropy(dist.predict);
  const div = tvDivergence(clip.predict.elite, clip.predict.field);
  const B = brier(pick.predict, pick.confidence, clip.outcome.actual, k);
  const Bc = brierChance(k);
  const skill = Math.round((1 - B / Bc) * 100);
  const correct = pick.predict === clip.outcome.actual;

  return (
    <div className="space-y-4 pt-2">
      <div style={{ background: PALETTE.panel, borderColor: PALETTE.line }} className="border p-4">
        <div className="flex items-center gap-2 mb-2">
          <Activity size={15} style={{ color: PALETTE.acid }} />
          <span className="text-xs uppercase tracking-widest" style={{ color: PALETTE.mute }}>Outcome</span>
          <span style={{ fontFamily: DISPLAY, color: clip.outcome.process === "SOUND" ? PALETTE.acid : clip.outcome.process === "UNSOUND" ? PALETTE.red : PALETTE.amber }} className="ml-auto text-sm tracking-wide">
            PROCESS: {clip.outcome.process}
          </span>
        </div>
        <div style={{ color: PALETTE.ink }} className="text-sm">{clip.outcome.summary}</div>
        {clip.outcome.keyFailure && (
          <div className="flex items-start gap-1.5 mt-2 text-sm" style={{ color: PALETTE.amber }}>
            <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {clip.outcome.keyFailure}
          </div>
        )}
      </div>

      <div className="flex gap-3">
        <Metric label="Disagreement" value={Hp.toFixed(2)} sub="norm. entropy [0–1]" color={Hp > 0.8 ? PALETTE.red : PALETTE.ink} />
        <Metric label="Top share" value={`${Math.round(topShare(dist.predict) * 100)}%`} sub="modal prediction" />
        <Metric label="Elite÷field" value={div.toFixed(2)} sub="TV divergence" color={PALETTE.acid} />
        <Metric label="Your skill" value={`${skill > 0 ? "+" : ""}${skill}`} sub={`Brier ${B.toFixed(2)} vs chance ${Bc.toFixed(2)}`} color={skill >= 0 ? PALETTE.acid : PALETTE.red} />
      </div>

      <div style={{ color: PALETTE.mute }} className="text-xs">
        {correct
          ? "Your prediction matched the realized outcome — but a single clip is not a verdict; reputation is your Brier aggregated over many clips."
          : "Your prediction missed this outcome. Outcomes are high-variance; a sound process can still lose a possession. Calibration emerges across clips, not within one."}
      </div>

      <div>
        <div className="text-xs uppercase tracking-widest mb-2" style={{ color: PALETTE.mute }}>Prediction — field distribution</div>
        <DistBar options={clip.predict.options} counts={dist.predict} highlight={pick.predict} actual={clip.outcome.actual} />
      </div>
      <div>
        <div className="text-xs uppercase tracking-widest mb-2" style={{ color: PALETTE.mute }}>Diagnosis — field distribution</div>
        <DistBar options={clip.diagnose.options} counts={dist.diagnose} highlight={pick.diagnose} actual={-1} />
      </div>
      <div>
        <div className="text-xs uppercase tracking-widest mb-2" style={{ color: PALETTE.mute }}>Adjustment — field distribution</div>
        <DistBar options={clip.adjust.options} counts={dist.adjust} highlight={pick.adjust} actual={-1} />
      </div>
    </div>
  );
}

/* ---------- feed ---------- */
function Feed({ clips, responses, onOpen }) {
  const ranked = useMemo(() => {
    return [...clips].map((c) => {
      const r = responses[c.id];
      const counts = c.predict.field.map((x, i) => x + (r && r.predict === i ? 1 : 0));
      return { clip: c, H: normEntropy(counts), N: counts.reduce((a, b) => a + b, 0), answered: !!r };
    }).sort((a, b) => b.H - a.H);
  }, [clips, responses]);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Zap size={16} style={{ color: PALETTE.acid }} />
        <span className="text-xs uppercase tracking-widest" style={{ color: PALETTE.mute }}>Most contested moments</span>
      </div>
      {ranked.map(({ clip, H, N, answered }) => (
        <button key={clip.id} onClick={() => onOpen(clip.id)}
          style={{ background: PALETTE.panel, borderColor: PALETTE.line }}
          className="border w-full p-4 text-left flex items-center gap-4">
          <div className="text-center w-14 shrink-0">
            <div style={{ fontFamily: DISPLAY, color: H > 0.8 ? PALETTE.red : PALETTE.acid }} className="text-3xl leading-none">{H.toFixed(2)}</div>
            <div className="text-xs" style={{ color: PALETTE.mute }}>disagree</div>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1"><Tag cat={clip.category} />
              {answered && <span className="text-xs flex items-center gap-1" style={{ color: PALETTE.acid }}><Check size={12} />answered</span>}
            </div>
            <div style={{ color: PALETTE.ink }} className="truncate">{clip.title}</div>
            <div className="text-xs mt-0.5" style={{ color: PALETTE.mute }}>{clip.context.score} · {clip.context.time} · {N} responses</div>
          </div>
          <ChevronRight size={18} style={{ color: PALETTE.mute }} />
        </button>
      ))}
    </div>
  );
}

/* ---------- leaderboard ---------- */
function Board({ userRep, answeredCount }) {
  const rows = useMemo(() => {
    const me = { name: "You", expert: false, rep: userRep, you: true };
    return [...SEED_BOARD, me].sort((a, b) => b.rep - a.rep);
  }, [userRep]);
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Trophy size={16} style={{ color: PALETTE.acid }} />
        <span className="text-xs uppercase tracking-widest" style={{ color: PALETTE.mute }}>Reputation — calibrated Brier skill, no consensus reward</span>
      </div>
      {rows.map((r, i) => (
        <div key={r.name} style={{ background: r.you ? PALETTE.panel2 : PALETTE.panel, borderColor: r.you ? PALETTE.acid : PALETTE.line }}
          className="border flex items-center gap-4 px-4 py-3">
          <span style={{ fontFamily: DISPLAY, color: PALETTE.mute }} className="text-2xl w-6">{i + 1}</span>
          <span style={{ color: PALETTE.ink }} className="flex-1">{r.name}
            {r.expert && <span style={{ color: PALETTE.acid }} className="text-xs ml-2 uppercase tracking-wide">expert</span>}
            {r.you && answeredCount === 0 && <span style={{ color: PALETTE.mute }} className="text-xs ml-2">(answer clips to rank)</span>}
          </span>
          <span style={{ fontFamily: DISPLAY, color: r.rep >= 0 ? PALETTE.acid : PALETTE.red }} className="text-2xl">{r.rep.toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------- root ---------- */
export default function ClipArena() {
  const [view, setView] = useState("feed");
  const [activeId, setActiveId] = useState(null);
  const [responses, setResponses] = useState({}); // clipId -> {predict,confidence,diagnose,adjust,brier}

  const answeredCount = Object.keys(responses).length;
  const userRep = useMemo(() => {
    const briers = Object.values(responses).map((r) => r.brier);
    if (!briers.length) return 0;
    const mean = briers.reduce((a, b) => a + b, 0) / briers.length;
    return 100 * (1 - mean / brierChance(5));
  }, [responses]);

  const active = CLIPS.find((c) => c.id === activeId);

  return (
    <div style={{ background: PALETTE.bg, color: PALETTE.ink, fontFamily: BODY, minHeight: "100vh" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Spline+Sans:wght@400;500;600&display=swap');`}</style>

      <div className="max-w-3xl mx-auto px-4 py-5">
        {/* header */}
        <div className="flex items-center justify-between mb-6 pb-4" style={{ borderBottom: `1px solid ${PALETTE.line}` }}>
          <div className="flex items-baseline gap-2">
            <span style={{ fontFamily: DISPLAY, color: PALETTE.acid }} className="text-3xl tracking-wider">CLIP ARENA</span>
            <span style={{ color: PALETTE.mute }} className="text-xs uppercase tracking-widest">Lacrosse IQ</span>
          </div>
          <div className="flex gap-1">
            {[["feed", "Feed"], ["board", "Board"]].map(([v, label]) => (
              <button key={v} onClick={() => setView(v)}
                style={{ borderColor: view === v ? PALETTE.acid : PALETTE.line, color: view === v ? PALETTE.acid : PALETTE.mute, fontFamily: DISPLAY }}
                className="border px-3 py-1 text-lg tracking-wide">{label}</button>
            ))}
          </div>
        </div>

        {view === "feed" && !active && (
          <Feed clips={CLIPS} responses={responses} onOpen={(id) => { setActiveId(id); }} />
        )}
        {view === "feed" && active && (
          <ClipView clip={active} prior={responses[active.id]}
            onBack={() => setActiveId(null)}
            onSubmit={(r) => setResponses((s) => ({ ...s, [active.id]: r }))} />
        )}
        {view === "board" && <Board userRep={userRep} answeredCount={answeredCount} />}

        <div className="mt-10 pt-4 text-xs" style={{ borderTop: `1px solid ${PALETTE.line}`, color: PALETTE.mute }}>
          Prototype · in-memory · seeded synthetic priors. Reputation is calibrated Brier skill vs chance — alignment with elite consensus is deliberately excluded from scoring (see design doc, R1).
        </div>
      </div>
    </div>
  );
}
