import { useState } from "react";

const MOCK_EMPLOYEES = [
  { id: "E001", name: "Sarah Chen", role: "Senior Data Scientist", dept: "Analytics", attrition: 87, salaryGap: -18.4, internalGap: -12.1, sentiment: "Frustrated", flag: "Salary + Career", plan: "Immediate 15% raise + Senior title review", managerNote: "Expressed frustration about promotion timeline in last 3 reviews. Recommend 1-on-1 career mapping session and fast-track to L6." },
  { id: "E002", name: "James Liu", role: "ML Engineer", dept: "Engineering", attrition: 79, salaryGap: -22.1, internalGap: -9.3, sentiment: "Disengaged", flag: "Workload + Salary", plan: "10% raise + workload rebalancing in Q3", managerNote: "Consistently rated high performer but recent signals suggest burnout. Consider redistributing 2 concurrent projects." },
  { id: "E003", name: "Maria Torres", role: "Product Manager", dept: "Product", attrition: 74, salaryGap: -8.2, internalGap: -14.7, sentiment: "Concerned", flag: "Management", plan: "Role expansion + equity refresh", managerNote: "Manager relationship scored 2.1/5 in anonymous survey. Escalation risk. Suggest skip-level meeting within 2 weeks." },
  { id: "E004", name: "Kevin Park", role: "Data Engineer", dept: "Data Platform", attrition: 68, salaryGap: -11.5, internalGap: -7.8, sentiment: "Neutral", flag: "Career", plan: "Promotion consideration in next cycle", managerNote: "Strong technical contributor. Career growth conversation overdue. Provide clear L4→L5 promotion criteria." },
  { id: "E005", name: "Aisha Johnson", role: "Research Scientist", dept: "AI Research", attrition: 61, salaryGap: -5.3, internalGap: -19.2, sentiment: "Concerned", flag: "Internal Equity", plan: "Internal equity adjustment $8K", managerNote: "Peer salary comparison shows significant underpay vs equivalent tenure. Silent flight risk. Address proactively." },
];

const CREDENTIALS = { hr: { user: "hr_admin", pass: "hr2026" }, manager: { user: "mgr_lead", pass: "mgr2026" } };
const riskColor = (v) => v >= 80 ? "#FF3B30" : v >= 65 ? "#FF9500" : "#34C759";
const riskLabel = (v) => v >= 80 ? "Critical" : v >= 65 ? "High" : "Moderate";

const GaugeRing = ({ value, size = 64 }) => {
  const r = (size - 10) / 2, circ = 2 * Math.PI * r, fill = (value / 100) * circ;
  return (
    <svg width={size} height={size} style={{ flexShrink: 0 }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#F2F2F7" strokeWidth={6}/>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={riskColor(value)} strokeWidth={6}
        strokeDasharray={`${fill} ${circ}`} strokeLinecap="round" transform={`rotate(-90 ${size/2} ${size/2})`}/>
      <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" fontSize="12" fontWeight="700" fill={riskColor(value)}>{value}%</text>
    </svg>
  );
};

const Badge = ({ text }) => {
  const map = { "Salary + Career": "#007AFF", "Workload + Salary": "#FF9500", "Management": "#FF3B30", "Career": "#34C759", "Internal Equity": "#5856D6" };
  const c = map[text] || "#8E8E93";
  return <span style={{ fontSize: 12, fontWeight: 600, color: c, background: c+"18", borderRadius: 6, padding: "3px 10px", whiteSpace: "nowrap" }}>{text}</span>;
};

const Avatar = ({ name, id }) => {
  const hue = id.charCodeAt(1) * 37 % 360;
  return <div style={{ width: 46, height: 46, borderRadius: "50%", background: `hsl(${hue},55%,88%)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, fontWeight: 700, color: `hsl(${hue},55%,38%)`, flexShrink: 0 }}>{name.split(" ").map(n=>n[0]).join("")}</div>;
};

const globalStyle = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body, #root { width: 100%; min-height: 100vh; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif; background: #F2F2F7; color: #1C1C1E; }
`;

export default function App() {
  const [screen, setScreen] = useState("home");
  const [loginForm, setLoginForm] = useState({ user: "", pass: "" });
  const [loginError, setLoginError] = useState("");
  const [selected, setSelected] = useState(null);
  const [loginTarget, setLoginTarget] = useState("");

  const go = (role) => { setLoginTarget(role); setLoginForm({ user:"", pass:"" }); setLoginError(""); setScreen("login"); };
  const login = () => {
    const c = CREDENTIALS[loginTarget];
    if (loginForm.user === c.user && loginForm.pass === c.pass) { setScreen(loginTarget); setSelected(null); }
    else setLoginError("Incorrect username or password.");
  };
  const toggle = (emp) => setSelected(s => s?.id === emp.id ? null : emp);
  const sorted = [...MOCK_EMPLOYEES].sort((a,b) => b.attrition - a.attrition);

  const NavBar = ({ title }) => (
    <div style={{ width: "100%", background: "rgba(255,255,255,0.88)", backdropFilter: "blur(24px)", borderBottom: "1px solid #E5E5EA", height: 56, display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 5%", position: "sticky", top: 0, zIndex: 100 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 13, color: "#8E8E93", fontWeight: 500 }}>RetentionAgent</span>
        <span style={{ color: "#D1D1D6" }}>/</span>
        <span style={{ fontSize: 14, fontWeight: 700 }}>{title}</span>
      </div>
      <button onClick={() => { setScreen("home"); setSelected(null); }} style={{ background: "none", border: "1.5px solid #E5E5EA", borderRadius: 10, padding: "6px 18px", fontSize: 13, color: "#636366", cursor: "pointer", fontFamily: "inherit" }}>Sign Out</button>
    </div>
  );

  const Card = ({ children, style = {}, onClick, onHover }) => (
    <div onClick={onClick}
      onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 6px 24px rgba(0,0,0,0.11)"; onHover?.(e, true); }}
      onMouseLeave={e => { e.currentTarget.style.boxShadow = "0 1px 4px rgba(0,0,0,0.07)"; onHover?.(e, false); }}
      style={{ background: "#fff", borderRadius: 18, padding: "22px 26px", boxShadow: "0 1px 4px rgba(0,0,0,0.07)", transition: "box-shadow 0.15s", ...style }}>
      {children}
    </div>
  );

  // ── HOME ─────────────────────────────────────────────────────
  if (screen === "home") return (
    <>
      <style>{globalStyle}</style>
      <div style={{ width: "100%", minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "48px 24px", background: "linear-gradient(160deg,#F2F2F7 0%,#FAFAFA 100%)" }}>
        <div style={{ textAlign: "center", marginBottom: 52 }}>
          <div style={{ display: "inline-block", background: "#007AFF", borderRadius: 12, padding: "8px 18px", fontSize: 11, fontWeight: 700, letterSpacing: 2, color: "#fff", textTransform: "uppercase", marginBottom: 18 }}>RetentionAgent</div>
          <div style={{ fontSize: "clamp(28px, 4vw, 48px)", fontWeight: 700, letterSpacing: -1, color: "#1C1C1E", lineHeight: 1.15 }}>People Intelligence Platform</div>
          <div style={{ fontSize: "clamp(14px, 1.5vw, 18px)", color: "#8E8E93", marginTop: 12 }}>AI-powered retention risk analysis for HR & Management</div>
        </div>

        <div style={{ display: "flex", gap: 24, flexWrap: "wrap", justifyContent: "center", width: "100%", maxWidth: 700 }}>
          {[
            { role: "hr", label: "HR Portal", sub: "Salary analytics & compensation planning", icon: "👥", desc: ["At-risk employee list", "Market & internal salary gaps", "Compensation adjustment plans"] },
            { role: "manager", label: "Manager View", sub: "Team retention insights", icon: "📊", desc: ["Team flight risk scores", "AI retention recommendations", "Action timeline guidance"] },
          ].map(({ role, label, sub, icon, desc }) => (
            <div key={role}
              onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-4px)"; e.currentTarget.style.boxShadow = "0 12px 36px rgba(0,0,0,0.13)"; }}
              onMouseLeave={e => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = "0 2px 20px rgba(0,0,0,0.08)"; }}
              style={{ background: "#fff", borderRadius: 24, padding: "36px 32px", flex: "1 1 260px", maxWidth: 320, boxShadow: "0 2px 20px rgba(0,0,0,0.08)", transition: "transform 0.18s, box-shadow 0.18s" }}>
              <div style={{ fontSize: 38, marginBottom: 16 }}>{icon}</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{label}</div>
              <div style={{ fontSize: 14, color: "#8E8E93", marginTop: 6, marginBottom: 20, lineHeight: 1.55 }}>{sub}</div>
              <div style={{ borderTop: "1px solid #F2F2F7", paddingTop: 16, marginBottom: 24 }}>
                {desc.map(d => <div key={d} style={{ fontSize: 13, color: "#636366", marginBottom: 8, display: "flex", gap: 8 }}><span style={{ color: "#34C759" }}>✓</span>{d}</div>)}
              </div>
              <button onClick={() => go(role)} style={{ width: "100%", padding: "13px 0", background: "#007AFF", color: "#fff", border: "none", borderRadius: 12, fontSize: 15, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>Sign In →</button>
            </div>
          ))}
        </div>
      </div>
    </>
  );

  // ── LOGIN ────────────────────────────────────────────────────
  if (screen === "login") return (
    <>
      <style>{globalStyle}</style>
      <div style={{ width: "100%", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "linear-gradient(160deg,#F2F2F7 0%,#FAFAFA 100%)", padding: 24 }}>
        <div style={{ background: "#fff", borderRadius: 28, padding: "44px 48px", width: "100%", maxWidth: 400, boxShadow: "0 4px 40px rgba(0,0,0,0.11)" }}>
          <button onClick={() => setScreen("home")} style={{ background: "none", border: "none", color: "#007AFF", fontSize: 14, cursor: "pointer", padding: 0, marginBottom: 28, fontFamily: "inherit" }}>← Back</button>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.8, color: "#8E8E93", textTransform: "uppercase", marginBottom: 8 }}>{loginTarget === "hr" ? "HR Portal" : "Manager Portal"}</div>
          <div style={{ fontSize: 30, fontWeight: 700, marginBottom: 32 }}>Sign In</div>
          {[{ f: "user", label: "Username", type: "text" }, { f: "pass", label: "Password", type: "password" }].map(({ f, label, type }) => (
            <div key={f} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#636366", marginBottom: 7 }}>{label}</div>
              <input type={type} value={loginForm[f]} onChange={e => setLoginForm(p => ({ ...p, [f]: e.target.value }))} onKeyDown={e => e.key === "Enter" && login()}
                onFocus={e => e.target.style.borderColor = "#007AFF"} onBlur={e => e.target.style.borderColor = "#E5E5EA"}
                style={{ width: "100%", padding: "13px 16px", borderRadius: 12, border: "1.5px solid #E5E5EA", fontSize: 15, outline: "none", fontFamily: "inherit", color: "#1C1C1E", transition: "border-color 0.15s" }}/>
            </div>
          ))}
          {loginError && <div style={{ fontSize: 13, color: "#FF3B30", marginBottom: 12, padding: "10px 14px", background: "#FFF0EE", borderRadius: 10 }}>{loginError}</div>}
          <button onClick={login} style={{ width: "100%", padding: "14px 0", background: "#007AFF", color: "#fff", border: "none", borderRadius: 13, fontSize: 16, fontWeight: 600, cursor: "pointer", marginTop: 8, fontFamily: "inherit" }}>Continue</button>
          <div style={{ fontSize: 12, color: "#C7C7CC", marginTop: 20, textAlign: "center" }}>Demo: <strong>{loginTarget === "hr" ? "hr_admin / hr2026" : "mgr_lead / mgr2026"}</strong></div>
        </div>
      </div>
    </>
  );

  // ── HR DASHBOARD ─────────────────────────────────────────────
  if (screen === "hr") {
    const criticalCount = MOCK_EMPLOYEES.filter(e => e.attrition >= 80).length;
    const avgGap = (MOCK_EMPLOYEES.reduce((s,e) => s + e.salaryGap, 0) / MOCK_EMPLOYEES.length).toFixed(1);
    return (
      <>
        <style>{globalStyle}</style>
        <div style={{ width: "100%", minHeight: "100vh" }}>
          <NavBar title="HR Dashboard" />
          <div style={{ width: "100%", maxWidth: 1200, margin: "0 auto", padding: "40px 5%" }}>
            <div style={{ marginBottom: 32 }}>
              <div style={{ fontSize: "clamp(24px, 2.5vw, 32px)", fontWeight: 700, letterSpacing: -0.5 }}>At-Risk Employees</div>
              <div style={{ fontSize: 15, color: "#8E8E93", marginTop: 6 }}>Sorted by attrition probability · Powered by RetentionAgent ML pipeline</div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, marginBottom: 28 }}>
              {[
                { label: "Monitored", value: MOCK_EMPLOYEES.length, unit: "employees", accent: "#007AFF" },
                { label: "Critical Risk", value: criticalCount, unit: "require immediate action", accent: "#FF3B30" },
                { label: "Avg Market Gap", value: avgGap + "%", unit: "below market median", accent: "#FF9500" },
              ].map(({ label, value, unit, accent }) => (
                <div key={label} style={{ background: "#fff", borderRadius: 18, padding: "22px 24px", boxShadow: "0 1px 4px rgba(0,0,0,0.07)" }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "#8E8E93", textTransform: "uppercase", letterSpacing: 1 }}>{label}</div>
                  <div style={{ fontSize: "clamp(28px, 3vw, 40px)", fontWeight: 700, color: accent, marginTop: 8, letterSpacing: -1 }}>{value}</div>
                  <div style={{ fontSize: 13, color: "#C7C7CC", marginTop: 4 }}>{unit}</div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {sorted.map(emp => (
                <Card key={emp.id} onClick={() => toggle(emp)} style={{ borderLeft: `5px solid ${riskColor(emp.attrition)}`, cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
                    <GaugeRing value={emp.attrition} size={64}/>
                    <div style={{ flex: 1, minWidth: 200 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                        <span style={{ fontWeight: 700, fontSize: 16 }}>{emp.name}</span>
                        <span style={{ fontSize: 13, color: "#8E8E93" }}>{emp.role} · {emp.dept}</span>
                        <Badge text={emp.flag}/>
                        <span style={{ fontSize: 12, fontWeight: 700, color: riskColor(emp.attrition), background: riskColor(emp.attrition)+"15", padding: "2px 10px", borderRadius: 6 }}>{riskLabel(emp.attrition)}</span>
                      </div>
                      <div style={{ display: "flex", gap: 24, marginTop: 8, flexWrap: "wrap" }}>
                        {[["Market Gap", emp.salaryGap+"%", "#FF3B30"], ["Internal Gap", emp.internalGap+"%", "#FF9500"], ["Sentiment", emp.sentiment, "#1C1C1E"]].map(([l,v,c]) => (
                          <div key={l} style={{ fontSize: 13, color: "#8E8E93" }}>{l}: <span style={{ color: c, fontWeight: 600 }}>{v}</span></div>
                        ))}
                      </div>
                    </div>
                    <span style={{ fontSize: 20, color: "#C7C7CC", transition: "transform 0.2s", transform: selected?.id === emp.id ? "rotate(90deg)" : "" }}>›</span>
                  </div>
                  {selected?.id === emp.id && (
                    <div style={{ marginTop: 22, paddingTop: 22, borderTop: "1px solid #F2F2F7", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 14 }}>
                      <div style={{ background: "#F9F9FB", borderRadius: 14, padding: "18px 20px" }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: "#8E8E93", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>💰 Compensation Plan</div>
                        <div style={{ fontSize: 14, lineHeight: 1.65 }}>{emp.plan}</div>
                      </div>
                      <div style={{ background: "#F9F9FB", borderRadius: 14, padding: "18px 20px" }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: "#8E8E93", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>📋 Risk Details</div>
                        {[["Department", emp.dept], ["Employee ID", emp.id], ["Attrition Score", emp.attrition+"%"]].map(([k,v]) => (
                          <div key={k} style={{ fontSize: 14, marginBottom: 4 }}>{k}: <strong style={k==="Attrition Score"?{color:riskColor(emp.attrition)}:{}}>{v}</strong></div>
                        ))}
                      </div>
                    </div>
                  )}
                </Card>
              ))}
            </div>
          </div>
        </div>
      </>
    );
  }

  // ── MANAGER DASHBOARD ────────────────────────────────────────
  if (screen === "manager") return (
    <>
      <style>{globalStyle}</style>
      <div style={{ width: "100%", minHeight: "100vh" }}>
        <NavBar title="Manager View" />
        <div style={{ width: "100%", maxWidth: 1200, margin: "0 auto", padding: "40px 5%" }}>
          <div style={{ marginBottom: 32 }}>
            <div style={{ fontSize: "clamp(24px, 2.5vw, 32px)", fontWeight: 700, letterSpacing: -0.5 }}>Team Retention Board</div>
            <div style={{ fontSize: 15, color: "#8E8E93", marginTop: 6 }}>AI-generated retention recommendations · Click any card to expand</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {sorted.map(emp => (
              <Card key={emp.id} onClick={() => toggle(emp)} style={{ cursor: "pointer" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                  <Avatar name={emp.name} id={emp.id}/>
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                      <span style={{ fontWeight: 700, fontSize: 16 }}>{emp.name}</span>
                      <span style={{ fontSize: 13, color: "#8E8E93" }}>{emp.role} · {emp.dept}</span>
                      <Badge text={emp.flag}/>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
                      <div style={{ height: 7, width: 160, background: "#F2F2F7", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${emp.attrition}%`, background: riskColor(emp.attrition), borderRadius: 4 }}/>
                      </div>
                      <span style={{ fontSize: 14, fontWeight: 700, color: riskColor(emp.attrition) }}>{emp.attrition}%</span>
                      <span style={{ fontSize: 13, color: "#8E8E93" }}>flight risk</span>
                      <span style={{ fontSize: 12, fontWeight: 600, color: riskColor(emp.attrition), background: riskColor(emp.attrition)+"15", padding: "2px 10px", borderRadius: 6 }}>{riskLabel(emp.attrition)}</span>
                    </div>
                  </div>
                  <span style={{ fontSize: 20, color: "#C7C7CC", transition: "transform 0.2s", transform: selected?.id === emp.id ? "rotate(90deg)" : "" }}>›</span>
                </div>
                {selected?.id === emp.id && (
                  <div style={{ marginTop: 22, paddingTop: 22, borderTop: "1px solid #F2F2F7", display: "flex", flexDirection: "column", gap: 12 }}>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
                      {[["Sentiment", emp.sentiment, null], ["Attrition Risk", emp.attrition+"%", riskColor(emp.attrition)], ["Primary Flag", emp.flag, null]].map(([l,v,c]) => (
                        <div key={l} style={{ background: "#F9F9FB", borderRadius: 12, padding: "14px 16px" }}>
                          <div style={{ fontSize: 11, color: "#8E8E93", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>{l}</div>
                          <div style={{ fontSize: 15, fontWeight: 700, marginTop: 6, color: c||"#1C1C1E" }}>{v}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{ background: "linear-gradient(135deg,#EEF5FF,#E5EFFF)", borderRadius: 14, padding: "18px 20px", border: "1px solid #C8DEFF" }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: "#007AFF", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 10 }}>🤖 AI Retention Recommendation</div>
                      <div style={{ fontSize: 14, lineHeight: 1.7 }}>{emp.managerNote}</div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ width: 9, height: 9, borderRadius: "50%", background: riskColor(emp.attrition), flexShrink: 0 }}/>
                      <span style={{ fontSize: 14, color: "#636366" }}>
                        {emp.attrition >= 80 ? "⚠️ Action required within 1 week" : emp.attrition >= 65 ? "Follow up within 2 weeks" : "Monitor in next review cycle"}
                      </span>
                    </div>
                  </div>
                )}
              </Card>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}