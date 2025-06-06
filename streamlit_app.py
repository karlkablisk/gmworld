# rp_ai_chat.py  –  FULL REVISION with all keys fixed!
# ---------------------------------------------------------------------------
# ▶ Run with:  streamlit run rp_ai_chat.py
# ---------------------------------------------------------------------------
import os, json, uuid, datetime, textwrap, base64
from pathlib import Path

import requests
import streamlit as st

# ========================== GLOBAL CONFIG ===================================
SAVE_ROOT         = Path("rp_saves")
CHAR_DIR          = SAVE_ROOT / "characters"
SCENE_DIR         = SAVE_ROOT / "scenes"
SESSION_DIR       = SAVE_ROOT / "sessions"
WORLD_FILE        = SAVE_ROOT / "world.json"
PERSONALITY_FILE  = "personalityprompt.txt"

OLLAMA_URL        = "http://localhost:11434"
A1111_URL         = "http://localhost:7860"

# ----------------------------------------------------------------------------
for p in (CHAR_DIR, SCENE_DIR, SESSION_DIR):
    p.mkdir(parents=True, exist_ok=True)

# =============  BACKEND (OLLAMA or OPENAI)  =================================
def get_backends():
    return ["ollama", "openai"]

st.sidebar.header("⚙️ AI Backend")
backend = st.sidebar.selectbox("Choose backend", get_backends(), key="backend")

if backend == "openai":
    import openai
    api_key = st.sidebar.text_input(
        "OpenAI API key", 
        type="password",
        value=st.session_state.get("openai_key", ""),
        key="openai_api_key"
    )
    if api_key:
        openai.api_key = api_key
        st.session_state.openai_key = api_key
else:
    api_key = None  # not used for Ollama

def chat_backend(model: str, messages: list[dict], temp: float = 0.7) -> str:
    """
    Unified chat wrapper for Ollama (local) or OpenAI (cloud).
    """
    if backend == "ollama":
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages,
                  "stream": False, "options": {"temperature": temp}},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    else:  # openai
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temp
        )
        return resp.choices[0].message.content

def list_models() -> list[str]:
    """
    Return model list depending on backend.
    """
    if backend == "ollama":
        try:
            tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
            return ["gemma3:latest"] + sorted({m["name"] for m in tags.get("models", [])})
        except Exception:
            return ["gemma3:latest"]
    else:
        # static shortlist for convenience; user can type any OpenAI model name
        return ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

MODELS_CACHE = list_models()

# ====================== UTILITY: LOAD / SAVE JSON ===========================
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# ============================ IMAGE GENERATOR ===============================
def generate_image(prompt: str, save_path: str) -> bool:
    """
    Simple txt2img call to Automatic1111. Returns True if success.
    """
    try:
        r = requests.post(f"{A1111_URL}/sdapi/v1/txt2img",
                          json={"prompt": prompt, "steps": 15, "width": 512, "height": 768},
                          timeout=120)
        r.raise_for_status()
        img_b64 = r.json()["images"][0]
        img_bytes = base64.b64decode(img_b64.split(",", 1)[-1])
        with open(save_path, "wb") as f:
            f.write(img_bytes)
        return True
    except Exception:
        return False

# ======================== STATE INITIALIZATION ==============================
if "init" not in st.session_state:
    st.session_state.init = True

    # Load world
    st.session_state.world = load_json(WORLD_FILE, {
        "prompt": "", "gm_system": "", "gm_notes": "", "gm_model": "gemma3:latest",
        "events": []
    })

    # Load characters & scenes
    st.session_state.characters = {p.stem: load_json(p, {}) for p in CHAR_DIR.glob("*.json")}
    st.session_state.scenes = {p.stem: load_json(p, {}) for p in SCENE_DIR.glob("*.json")}

    # Session variables
    st.session_state.current_session = None
    st.session_state.chat_log = []
    st.session_state.timeline = []
    st.session_state.gm_history = []
    st.session_state.char_history = {}
    st.session_state.active_scene = None
    st.session_state.player_name = "PLAYER"
    st.session_state.player_char = {}
    st.session_state.show_thoughts = True

# --------------------- LOAD LONG PERSONALITY TEMPLATE -----------------------
PERSONALITY_TEMPLATE = load_json(Path(PERSONALITY_FILE), "") \
    if PERSONALITY_FILE.endswith(".json") else Path(PERSONALITY_FILE).read_text(encoding="utf-8") \
    if Path(PERSONALITY_FILE).exists() else "TEMPLATE NOT FOUND"

# =========================== SIDEBAR UI =====================================

# -------- Session Picker & Export / Import ----------------------------------
st.sidebar.header("🗂 Session")
sess_files = ["<new>"] + sorted(f.name for f in SESSION_DIR.glob("*.json"))
sel_sess = st.sidebar.selectbox("Current session", sess_files, key="sess_selectbox")

def new_session():
    fname = f"session_{datetime.date.today()}_{uuid.uuid4().hex[:6]}.json"
    st.session_state.current_session = SESSION_DIR / fname
    st.session_state.chat_log, st.session_state.timeline = [], []
    st.session_state.gm_history, st.session_state.char_history = [], {}
def load_session(fname):
    data = load_json(SESSION_DIR / fname, {"chat_log": [], "timeline": [],
                                           "gm_history": [], "char_history": {}})
    st.session_state.chat_log = data["chat_log"]
    st.session_state.timeline = data["timeline"]
    st.session_state.gm_history = data["gm_history"]
    st.session_state.char_history = data["char_history"]
    st.session_state.current_session = SESSION_DIR / fname
if sel_sess == "<new>":
    new_session()
else:
    load_session(sel_sess)
st.sidebar.write(f"File: {st.session_state.current_session.name}")

# Export / Import buttons
exp_col, imp_col = st.sidebar.columns(2)
with exp_col:
    if st.button("Export ▶️", key="export_btn"):
        st.session_state.export_json = json.dumps({
            "world": st.session_state.world,
            "characters": st.session_state.characters,
            "scenes": st.session_state.scenes,
            "session": {
                "chat_log": st.session_state.chat_log,
                "timeline": st.session_state.timeline,
                "gm_history": st.session_state.gm_history,
                "char_history": st.session_state.char_history
            }
        }, indent=2)
with imp_col:
    if st.button("⬅️ Import", key="import_btn"):
        imp_str = st.text_area("Paste JSON to import and click 'Load import'", key="import_textarea").strip()
        if imp_str and st.button("Load import", key="load_import_btn"):
            data = json.loads(imp_str)
            st.session_state.world = data.get("world", st.session_state.world)
            st.session_state.characters = data.get("characters", st.session_state.characters)
            st.session_state.scenes = data.get("scenes", st.session_state.scenes)
            s = data.get("session", {})
            st.session_state.chat_log = s.get("chat_log", [])
            st.session_state.timeline = s.get("timeline", [])
            st.session_state.gm_history = s.get("gm_history", [])
            st.session_state.char_history = s.get("char_history", {})
            st.success("Imported.")

# ---------- World / GM Settings --------------------------------------------
st.sidebar.header("🌍 World & GM")
w = st.session_state.world
w["prompt"]     = st.sidebar.text_area("World prompt", w["prompt"], height=90, key="world_prompt")
w["gm_system"]  = st.sidebar.text_area("GM system prompt", w["gm_system"], height=70, key="gm_system_prompt")
w["gm_notes"]   = st.sidebar.text_area("GM notes / plot", w["gm_notes"], height=70, key="gm_notes")
w["gm_model"]   = st.sidebar.selectbox("GM model", MODELS_CACHE,
                                       index=MODELS_CACHE.index(w["gm_model"])
                                       if w["gm_model"] in MODELS_CACHE else 0,
                                       key="gm_model_select")

if st.sidebar.button("Save World", key="save_world_btn"):
    save_json(WORLD_FILE, w)
    st.toast("World saved")

# ------------ Events --------------------------------------------------------
st.sidebar.subheader("📜 Events")
ev_names = ["<new>"] + [e["name"] for e in w["events"]]
sel_ev = st.sidebar.selectbox("Select event", ev_names, key="event_selectbox")
if sel_ev == "<new>":
    ev = {"id": uuid.uuid4().hex, "name": "", "scene": "", "desc": "",
          "time": datetime.datetime.now().isoformat(timespec="seconds")}
else:
    ev = next(e for e in w["events"] if e["name"] == sel_ev)
ev["name"]  = st.sidebar.text_input("Title", ev["name"], key="event_title")
ev["scene"] = st.sidebar.text_input("Scene", ev["scene"], key="event_scene")
ev["desc"]  = st.sidebar.text_area("Description", str(ev.get("desc") or ""), height=70, key="event_desc")

if st.sidebar.button("Save event", key="save_event_btn"):
    if sel_ev == "<new>":
        w["events"].append(ev)
    save_json(WORLD_FILE, w)
    st.toast("Event saved")

# ------------ Scene Manager -------------------------------------------------
st.sidebar.header("🎬 Scene")
scene_names = ["<new>"] + sorted(st.session_state.scenes)
sel_scene = st.sidebar.selectbox("Active scene", scene_names,
                                 index=scene_names.index(st.session_state.active_scene)
                                 if st.session_state.active_scene in scene_names else 0,
                                 key="scene_selectbox")
if sel_scene == "<new>":
    scene = {"name": "", "prompt": "", "image": "", "chars": []}
else:
    scene = st.session_state.scenes[sel_scene].copy()

scene["name"]   = st.sidebar.text_input("Scene name", scene["name"], key="scene_name")
scene["prompt"] = st.sidebar.text_area("Scene description", scene.get("prompt") or "", height=70, key="scene_prompt")

# Image
scene["image"]  = st.sidebar.text_input("Image path", scene.get("image", ""), key="scene_image_path")
if scene["image"] and Path(scene["image"]).exists():
    st.sidebar.image(scene["image"], caption="Scene image", use_column_width=True)
img_prompt = st.sidebar.text_input("Prompt for scene image", key="scene_img_prompt")
if st.sidebar.button("Generate scene image", key="gen_scene_img_btn"):
    ok = generate_image(img_prompt, scene["image"] or f"{scene['name']}_scene.png")
    st.toast("Generated" if ok else "Failed")
# Characters in scene
st.sidebar.write("**Chars in scene**")
for c in scene["chars"]:
    st.sidebar.write("•", c)
add_c = st.sidebar.text_input("Add char by name", key="add_char_by_name")
if st.sidebar.button("Add char", key="add_char_btn"):
    if add_c and add_c not in scene["chars"]:
        scene["chars"].append(add_c)
# Mass auto-generate missing chars
if st.sidebar.button("Auto-generate missing", key="autogen_chars_btn"):
    missing = [c for c in scene["chars"] if c not in st.session_state.characters]
    for cname in missing:
        quick = f"Create a concise ID sheet for {cname} suitable to the scene {scene['name']}."
        text  = chat_backend(w["gm_model"], [{"role": "user", "content": quick}], 0.3)
        st.session_state.characters[cname] = {"name": cname, "race": "", "gender": "",
                                              "model": w["gm_model"], "tts": "",
                                              "personality": text, "prompt_tweak": "",
                                              "develop": "", "in_scene": True,
                                              "image": "", "party": False}
        save_json(CHAR_DIR / f"{cname}.json", st.session_state.characters[cname])
    st.toast(f"Generated {len(missing)} chars")

# Scene suggestion
if st.sidebar.button("Suggest next scene", key="suggest_scene_btn"):
    sugg = chat_backend(w["gm_model"], [
        {"role": "system", "content": w["gm_system"]},
        {"role": "user", "content": "Suggest a compelling next scene: output as JSON with keys name, prompt, suggested_chars"}
    ], 0.7)
    st.sidebar.code(sugg, language="json", key="scene_suggestion_code")

if st.sidebar.button("Save scene", key="save_scene_btn"):
    st.session_state.scenes[scene["name"]] = scene
    save_json(SCENE_DIR / f"{scene['name']}.json", scene)
    st.session_state.active_scene = scene["name"]
    st.toast("Scene saved")

# Update in_scene flags automatically (+ party)
for cname, c in st.session_state.characters.items():
    if c.get("party"):
        c["in_scene"] = True  # party members always follow
    else:
        c["in_scene"] = cname in scene["chars"]

# -------------- Character Editor --------------------------------------------
st.sidebar.header("👥 Characters")
chars_list = ["<new>"] + sorted(st.session_state.characters)
sel_char = st.sidebar.selectbox("Edit char", chars_list, key="edit_char_selectbox")
if sel_char == "<new>":
    c = {"name": "", "race": "", "gender": "", "model": MODELS_CACHE[0], "tts": "",
         "personality": "", "prompt_tweak": "", "develop": "", "image": "",
         "in_scene": False, "party": False}
else:
    c = st.session_state.characters[sel_char].copy()

c["name"]  = st.sidebar.text_input("Name", c["name"], key="char_name")
c["race"]  = st.sidebar.text_input("Race", c["race"], key="char_race")
c["gender"]= st.sidebar.text_input("Gender", c["gender"], key="char_gender")
c["model"] = st.sidebar.selectbox("Model", MODELS_CACHE,
                                  index=MODELS_CACHE.index(c["model"])
                                  if c["model"] in MODELS_CACHE else 0,
                                  key="char_model_select")
c["tts"]   = st.sidebar.text_input("TTS voice", c["tts"], key="char_tts")
c["image"] = st.sidebar.text_input("Image path", c.get("image", ""), key="char_image_path")
if c["image"] and Path(c["image"]).exists():
    st.sidebar.image(c["image"], use_column_width=True)
# Image gen
img_p = st.sidebar.text_input("Prompt for char image", key="char_img_prompt")
if st.sidebar.button("Generate char image", key="gen_char_img_btn"):
    ok = generate_image(img_p, c["image"] or f"{c['name']}_char.png")
    st.toast("Generated" if ok else "Failed")
c["personality"]   = st.sidebar.text_area("Personality", c["personality"], height=100, key="char_personality")
c["prompt_tweak"]  = st.sidebar.text_area("Prompt tweak", c["prompt_tweak"], height=70, key="char_prompt_tweak")
c["develop"]       = st.sidebar.text_area("Develop", c["develop"], height=70, key="char_develop")
c["party"]         = st.sidebar.checkbox("Party member", c.get("party", False), key="char_party")
c["in_scene"]      = st.sidebar.checkbox("Currently in scene", c.get("in_scene", False), key="char_in_scene")

# Save / Generate
if st.sidebar.button("Save char", key="save_char_btn"):
    if c["name"]:
        st.session_state.characters[c["name"]] = c
        save_json(CHAR_DIR / f"{c['name']}.json", c)
        st.toast("Saved")

if st.sidebar.button("Generate missing fields", key="gen_missing_fields_btn"):
    seed = textwrap.dedent(f"Name:{c['name']}\nRace:{c['race']}\nGender:{c['gender']}\n{c['prompt_tweak']}")
    result = chat_backend(c["model"], [{"role": "user", "content": seed + "\n\n" + PERSONALITY_TEMPLATE}], 0.3)
    c["personality"] = result
    st.session_state.characters[c["name"]] = c
    save_json(CHAR_DIR / f"{c['name']}.json", c)
    st.experimental_rerun()

# ----------------- Thought Log & Relationship Viewer ------------------------
st.sidebar.header("🧠 Thoughts / Relationships")
sel_view_char = st.sidebar.selectbox("View thoughts of", sorted(st.session_state.characters), key="thought_view_selectbox")
if sel_view_char:
    t_path = CHAR_DIR / f"{sel_view_char}_thoughts.json"
    t_data = load_json(t_path, {"opinions": {}, "events": []})
    st.sidebar.subheader("Recent Thoughts")
    for e in reversed(t_data["events"][-5:]):
        st.sidebar.write(f"- {e['time'][5:16]}: {e['text']}")
    st.sidebar.subheader("Opinions")
    for target, txt in t_data["opinions"].items():
        st.sidebar.write(f"**{target}** ➜ {txt.splitlines()[-1][:120]}…")

# Relationship map (simple matrix)
if st.sidebar.button("Show relationship table", key="show_rel_btn"):
    import pandas as pd
    chars = sorted(st.session_state.characters)
    mat = pd.DataFrame("", index=chars, columns=chars)
    for cname in chars:
        td = load_json(CHAR_DIR / f"{cname}_thoughts.json", {"opinions": {}})
        for tgt, txt in td["opinions"].items():
            mat.loc[cname, tgt] = txt.splitlines()[-1][:40]
    st.sidebar.dataframe(mat)

# ============================ MAIN ==========================================
st.title("🎭 Modular Role-Play")
st.checkbox("Show thoughts", key="show_thoughts")

# Timeline
st.subheader(f"Scene ▸ {st.session_state.active_scene or 'None'}")
for ev in reversed(st.session_state.timeline[-50:]):
    t = ev["time"][11:19] if "time" in ev else ""
    st.markdown(f"- **[{t}] {ev['type'].upper()}** – `{ev['actor']}`: {ev['content']}")

# Chat log
st.markdown("---")
for m in st.session_state.chat_log[-200:]:
    role, content = m["role"], m["content"]
    if not st.session_state.show_thoughts:
        content = "\n".join(l for l in content.splitlines()
                            if not l.lower().startswith("thought:"))
    img = st.session_state.characters.get(role, {}).get("image", "")
    if img and Path(img).exists():
        st.image(img, width=60)
    st.markdown(f"**{role}**:\n{content}")

# Export TTS
if st.button("Export TTS", key="export_tts_btn"):
    tts_lines = []
    for m in st.session_state.chat_log:
        for l in m["content"].splitlines():
            if l.lower().startswith("speech:"):
                tts_lines.append(f"{m['role']}: {l.split(':',1)[1].strip()}")
    st.text_area("TTS", value="\n".join(tts_lines), height=200, key="tts_export_textarea")

# =================== DEVELOPMENT & THOUGHT STORAGE ==========================
def auto_update_developments(narration: str):
    for cname, c in st.session_state.characters.items():
        if not c.get("in_scene"): continue
        summ = chat_backend(
            c["model"],
            [{"role": "system", "content": "You summarize personal changes (≤30 words)."},
             {"role": "user", "content": f"Previous develop:\n{c['develop']}\n\nScene:\n{narration}"}],
            0.3
        )
        c["develop"] = (c["develop"] + "\n" + summ).strip()[-800:]
        save_json(CHAR_DIR / f"{cname}.json", c)

def store_thought(cname: str, thought: str):
    tp = CHAR_DIR / f"{cname}_thoughts.json"
    data = load_json(tp, {"opinions": {}, "events": []})
    # simple heuristic: "about <Target>:" to store as opinion
    if thought.lower().startswith("about "):
        tgt, rest = thought[6:].split(":", 1)
        data["opinions"][tgt.strip()] = (data["opinions"].get(tgt.strip(),"") + "\n" + rest.strip())[-400:]
    else:
        data["events"].append({"time": datetime.datetime.now().isoformat(), "text": thought})
        data["events"] = data["events"][-60:]
    save_json(tp, data)

# ============================== TURN LOGIC ==================================
def run_turn(player_msg: str):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    if player_msg.strip():
        st.session_state.chat_log.append({"role": st.session_state.player_name,
                                          "content": f"Speech: {player_msg}"})
        st.session_state.timeline.append({"type": "player", "actor": st.session_state.player_name,
                                          "content": player_msg, "time": ts})

    # ----- GM NARRATION -----
    gm_msgs = [
        {"role": "system", "content": w["gm_system"]},
        {"role": "system", "content": w["prompt"]},
        {"role": "system", "content": scene["prompt"]},
        {"role": "system", "content": w["gm_notes"]},
        *st.session_state.gm_history[-10:],
        {"role": "user", "content": player_msg or "[continue]"}
    ]
    narration = chat_backend(w["gm_model"], gm_msgs)
    st.session_state.gm_history.append({"role": "assistant", "content": narration})
    st.session_state.chat_log.append({"role": "NARRATOR", "content": narration})
    st.session_state.timeline.append({"type": "narration", "actor": "NARRATOR",
                                      "content": narration, "time": ts})

    # ----- NPC REACTIONS -----
    for cname, c in st.session_state.characters.items():
        if not c.get("in_scene"): continue
        if cname == st.session_state.player_name: continue
        hist = st.session_state.char_history.setdefault(cname, [])
        sys_prompt = f"You are {cname}. Personality:\n{c['personality']}\nDevelop:\n{c['develop']}"
        ask = f"GM narration:\n{narration}\nRespond with:\nThought: ...\nSpeech: ..."
        reply = chat_backend(c["model"], [
            {"role": "system", "content": sys_prompt},
            *hist[-10:], {"role": "user", "content": ask}
        ])
        hist.append({"role": "assistant", "content": reply})
        st.session_state.chat_log.append({"role": cname, "content": reply})
        st.session_state.timeline.append({"type": "character", "actor": cname,
                                          "content": reply, "time": ts})
        for line in reply.splitlines():
            if line.lower().startswith("thought:"):
                store_thought(cname, line.split(":", 1)[1].strip())

    auto_update_developments(narration)

    # Save session
    save_json(st.session_state.current_session, {
        "chat_log": st.session_state.chat_log,
        "timeline": st.session_state.timeline,
        "gm_history": st.session_state.gm_history,
        "char_history": st.session_state.char_history
    })

# ================================ INPUT =====================================
player_input = st.chat_input("Your action / dialogue…", key="player_chat_input")
if player_input is not None:
    run_turn(player_input)
    st.experimental_rerun()
