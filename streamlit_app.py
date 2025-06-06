# rp_ai_chat.py  –  sidebar tabs, auto-TTS, world/event generation,
# backend-aware model lists, unique keys everywhere
# ---------------------------------------------------------------
# ▶ streamlit run rp_ai_chat.py
# ---------------------------------------------------------------
import os, json, uuid, datetime, textwrap, base64, itertools
from pathlib import Path
import requests
import streamlit as st

# -------------------- PATHS / CONSTANTS ------------------------
SAVE_ROOT = Path("rp_saves")
CHAR_DIR, SCENE_DIR, SESSION_DIR = (SAVE_ROOT / p for p in ("characters", "scenes", "sessions"))
WORLD_FILE = SAVE_ROOT / "world.json"
PERSONALITY_FILE = "personalityprompt.txt"
TTS_FILE = SAVE_ROOT / "tts_out.txt"

OLLAMA_URL = "http://localhost:11434"
A1111_URL = "http://localhost:7860"

for p in (CHAR_DIR, SCENE_DIR, SESSION_DIR): p.mkdir(parents=True, exist_ok=True)
SAVE_ROOT.mkdir(exist_ok=True)

# -------------------- BACKEND SELECTION ------------------------
st.sidebar.header("⚙️  Settings")
backend = st.sidebar.selectbox("AI backend", ["ollama", "openai"], key="backend_select")
def fetch_models():
    if backend == "ollama":
        try:
            tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=8).json()
            return ["gemma3:latest"] + sorted({m["name"] for m in tags.get("models", [])})
        except Exception:
            return ["gemma3:latest"]
    else:
        return ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
MODELS_CACHE = fetch_models()

# OpenAI key if needed
if backend == "openai":
    import openai
    openai.api_key = st.sidebar.text_input("OpenAI API key", type="password",
                                           value=st.session_state.get("openai_key",""),
                                           key="openai_key")
    st.session_state.openai_key = openai.api_key

# -------------------- CHAT WRAPPER -----------------------------
def chat_backend(model: str, messages: list[dict], temp: float = 0.7) -> str:
    if backend == "ollama":
        r = requests.post(f"{OLLAMA_URL}/api/chat",
                          json={"model":model,"messages":messages,"stream":False,
                                "options":{"temperature":temp}}, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    else:
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temp)
        return resp.choices[0].message.content

# -------------------- LOAD / SAVE HELPERS ----------------------
def load_json(path: Path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default
def save_json(path: Path, obj): path.write_text(json.dumps(obj,indent=2), encoding="utf-8")

# -------------------- STATE INIT -------------------------------
if "init" not in st.session_state:
    st.session_state.init=True
    st.session_state.world = load_json(WORLD_FILE, {"prompt":"","gm_system":"","gm_notes":"",
                                                    "gm_model":MODELS_CACHE[0],"events":[]})
    st.session_state.characters = {p.stem: load_json(p,{}) for p in CHAR_DIR.glob("*.json")}
    st.session_state.scenes     = {p.stem: load_json(p,{}) for p in SCENE_DIR.glob("*.json")}
    st.session_state.current_session=None
    st.session_state.chat_log=[]; st.session_state.timeline=[]
    st.session_state.gm_hist=[];  st.session_state.char_hist={}
    st.session_state.active_scene=None
    st.session_state.player_name="PLAYER"
    st.session_state.show_thoughts=True

# personality template
PERSONALITY_TEMPLATE = Path(PERSONALITY_FILE).read_text(encoding="utf-8") \
    if Path(PERSONALITY_FILE).exists() else "TEMPLATE NOT FOUND"

# -------------------- TABS LAYOUT (SIDEBAR) --------------------
tab_settings, tab_world, tab_scene, tab_char, tab_event, tab_thought = st.sidebar.tabs(
    ["⚙️ Settings", "🌍 World", "🎬 Scene", "👥 Characters", "📜 Events", "🧠 Thoughts"]
)

# ========== SETTINGS TAB (already holding backend above) =======
with tab_settings:
    st.checkbox("Show thoughts in chat", key="show_thoughts")

# ========== WORLD TAB ==========================================
with tab_world:
    w = st.session_state.world
    w["prompt"]    = st.text_area("World prompt", w["prompt"], height=90, key="world_prompt")
    w["gm_system"] = st.text_area("GM system prompt", w["gm_system"], height=70, key="gm_sys")
    w["gm_notes"]  = st.text_area("GM notes / plot", w["gm_notes"], height=70, key="gm_notes")
    # dynamic model list
    if w["gm_model"] not in MODELS_CACHE: MODELS_CACHE.append(w["gm_model"])
    w["gm_model"]  = st.selectbox("GM model", MODELS_CACHE,
                                  index=MODELS_CACHE.index(w["gm_model"]), key="gm_model")
    cols = st.columns(2)
    if cols[0].button("💾", help="Save world", key="save_world"):
        save_json(WORLD_FILE, w); st.toast("World saved")
    if cols[1].button("✨", help="Generate / fill missing", key="gen_world"):
        theme = w["prompt"] or "Generate a fresh imaginative setting"
        filled = chat_backend(w["gm_model"], [{"role":"user","content":
                 f"Create a concise world prompt, gm system prompt and 3-line gm_notes for: {theme}. "
                 "Return JSON with keys prompt, gm_system, gm_notes."}],0.5)
        try:
            new = json.loads(filled)
            w.update({k:new.get(k,w[k]) for k in ("prompt","gm_system","gm_notes")})
            st.toast("World generated")
        except Exception:
            st.error("Generation failed")

# ========== SESSION (top of Scene tab) =========================
def new_session():
    fname=f"session_{datetime.date.today()}_{uuid.uuid4().hex[:6]}.json"
    st.session_state.current_session=SESSION_DIR/fname
    st.session_state.chat_log=[]; st.session_state.timeline=[]
    st.session_state.gm_hist=[]; st.session_state.char_hist={}
def load_session(fname):
    data=load_json(SESSION_DIR/fname,{"chat_log":[],"timeline":[],"gm_hist":[],"char_hist":{}})
    st.session_state.chat_log=data["chat_log"]; st.session_state.timeline=data["timeline"]
    st.session_state.gm_hist=data["gm_hist"];   st.session_state.char_hist=data["char_hist"]
    st.session_state.current_session=SESSION_DIR/fname

# ========== SCENE TAB ==========================================
with tab_scene:
    sess_files=["<new>"]+sorted(p.name for p in SESSION_DIR.glob("*.json"))
    sel_sess=st.selectbox("Session file", sess_files, key="sess_file")
    if sel_sess=="<new>": new_session()
    else: load_session(sel_sess)
    scene_names=["<new>"]+sorted(st.session_state.scenes)
    sel_scene = st.selectbox("Active scene", scene_names, key="scene_select",
                             index=scene_names.index(st.session_state.active_scene)
                             if st.session_state.active_scene in scene_names else 0)
    if sel_scene=="<new>": scene={"name":"","prompt":"","image":"","chars":[]}
    else: scene=st.session_state.scenes[sel_scene].copy()
    scene["name"]   = st.text_input("Scene name", scene["name"], key="scene_name")
    scene["prompt"] = st.text_area("Scene description", scene.get("prompt",""), height=70, key="scene_prompt")
    scene["image"]  = st.text_input("Image path", scene.get("image",""), key="scene_img")
    if scene["image"] and Path(scene["image"]).exists(): st.image(scene["image"], use_column_width=True)
    imgpr = st.text_input("Prompt → generate image", key="scene_imgpr")
    if st.button("🖼️", key="gen_scene_img", help="Generate scene image"):
        ok=generate_image(imgpr, scene["image"] or f"{scene['name']}_scene.png")
        st.toast("OK" if ok else "Fail")
    st.markdown("**Characters in scene**")
    for c in scene["chars"]: st.write("•", c)
    add_c=st.text_input("Add char", key="addchar"); 
    st.button("➕",key="addchar_btn",help="Add",on_click=lambda: scene["chars"].append(add_c) if add_c and add_c not in scene["chars"] else None)
    st.button("Auto-generate missing", key="autogen_scene_chars",
              on_click=lambda: [(
                  lambda cname: (
                      save_json(CHAR_DIR/f"{cname}.json",
                                st.session_state.characters.setdefault(
                                    cname,{"name":cname,"model":w["gm_model"],
                                           "personality":chat_backend(w["gm_model"],
                                           [{"role":"user","content":f'Create a quick sheet for {cname}.'}],0.3),
                                           "race":"","gender":"","tts":"","prompt_tweak":"",
                                           "develop":"","image":"","in_scene":True,"party":False})))
                  )(c) for c in scene["chars"] if c not in st.session_state.characters])
    cols=st.columns(2)
    if cols[0].button("💾", key="save_scene"): 
        st.session_state.scenes[scene["name"]]=scene
        save_json(SCENE_DIR/f"{scene['name']}.json", scene)
        st.session_state.active_scene=scene["name"]; st.toast("Scene saved")
    if cols[1].button("✨", key="suggest_scene", help="AI suggest next scene"):
        js=chat_backend(w["gm_model"],[{"role":"system","content":w["gm_system"]},
           {"role":"user","content":"Suggest next scene as JSON {name,prompt,chars[]}"}])
        st.code(js,language="json")
    # update in_scene flags
    for cn,c in st.session_state.characters.items():
        c["in_scene"]=c.get("party") or cn in scene["chars"]

# ========== CHARACTER TAB ======================================
with tab_char:
    chars = ["<new>"]+sorted(st.session_state.characters)
    sel=st.selectbox("Select", chars, key="char_select")
    if sel=="<new>":
        c={"name":"","race":"","gender":"","model":MODELS_CACHE[0],"tts":"","personality":"",
           "prompt_tweak":"","develop":"","image":"","in_scene":False,"party":False}
    else: c=st.session_state.characters[sel].copy()
    left,right = st.columns(2)
    c["name"]= left.text_input("Name", c["name"], key="char_name")
    c["race"]= left.text_input("Race", c["race"], key="char_race")
    c["gender"]= left.text_input("Gender", c["gender"], key="char_gender")
    if c["model"] not in MODELS_CACHE: MODELS_CACHE.append(c["model"])
    c["model"]= left.selectbox("Model", MODELS_CACHE, index=MODELS_CACHE.index(c["model"]), key="char_model")
    c["tts"]= left.text_input("TTS voice", c["tts"], key="char_tts")
    c["party"]= left.checkbox("Party member", c.get("party",False), key="char_party")
    c["image"]= right.text_input("Image path", c.get("image",""), key="char_imgpath")
    if c["image"] and Path(c["image"]).exists(): right.image(c["image"],use_column_width=True)
    imgpr_char= right.text_input("Prompt → image", key="char_imgpr")
    if right.button("🖼️", key="gen_char_img"): 
        ok=generate_image(imgpr_char, c["image"] or f"{c['name']}_char.png"); st.toast("OK" if ok else "Fail")
    c["personality"]=st.text_area("Personality", c["personality"], height=100, key="char_pers")
    c["prompt_tweak"]=st.text_area("Prompt tweak", c["prompt_tweak"], height=70, key="char_ptweak")
    c["develop"]=st.text_area("Develop", c["develop"], height=70, key="char_dev")
    c["in_scene"]=st.checkbox("In current scene", c.get("in_scene",False), key="char_inscene")
    cols=st.columns(3)
    if cols[0].button("💾", key="save_char"):
        if c["name"]: save_json(CHAR_DIR/f"{c['name']}.json",c); st.session_state.characters[c["name"]]=c; st.toast("Saved")
    if cols[1].button("✨", key="gen_fields", help="Generate/fill"):
        seed=textwrap.dedent(f"Name:{c['name']}\nRace:{c['race']}\nGender:{c['gender']}\n{c['prompt_tweak']}")
        c["personality"]=chat_backend(c["model"],[{"role":"user","content":seed+"\n\n"+PERSONALITY_TEMPLATE}],0.3)
        st.experimental_rerun()
    if cols[2].button("🗑️",help="Delete",key="del_char") and sel!="<new>":
        (CHAR_DIR/f"{sel}.json").unlink(missing_ok=True); st.session_state.characters.pop(sel,None); st.experimental_rerun()

# ========== EVENTS TAB =========================================
with tab_event:
    ev_names=["<new>"]+[e["name"] for e in w["events"]]
    sel_ev=st.selectbox("Event", ev_names, key="ev_select")
    if sel_ev=="<new>": ev={"id":uuid.uuid4().hex,"name":"","scene":"","desc":"",
                           "time":datetime.datetime.now().isoformat(timespec="seconds")}
    else: ev=next(e for e in w["events"] if e["name"]==sel_ev)
    ev["name"]=st.text_input("Title", ev["name"], key="ev_title")
    ev["scene"]=st.text_input("Scene", ev["scene"], key="ev_scene")
    ev["desc"]=st.text_area("Description", ev.get("desc",""), height=70, key="ev_desc")
    cols=st.columns(3)
    if cols[0].button("💾", key="save_ev"): 
        if sel_ev=="<new>": w["events"].append(ev)
        save_json(WORLD_FILE,w); st.toast("Event saved")
    if cols[1].button("✨", key="gen_ev"):
        seed=ev["scene"] or w["prompt"] or "Random"
        filled=chat_backend(w["gm_model"],[{"role":"user","content":f"Write an interesting world event in <70 words for: {seed}. Return only description."}],0.5)
        ev["desc"]=filled; st.experimental_rerun()
    if cols[2].button("🗑️",key="del_ev") and sel_ev!="<new>":
        w["events"]=[e for e in w["events"] if e["id"]!=ev["id"]]; save_json(WORLD_FILE,w); st.experimental_rerun()

# ========== THOUGHTS TAB =======================================
with tab_thought:
    sel_view=st.selectbox("Character", sorted(st.session_state.characters), key="thought_char_sel")
    tp=CHAR_DIR/f"{sel_view}_thoughts.json"; tdat=load_json(tp,{"opinions":{},"events":[]})
    st.subheader("Recent thoughts")
    for e in reversed(tdat["events"][-5:]): st.write(f"- {e['time'][11:16]} {e['text']}")
    st.subheader("Opinions")
    for tgt,txt in tdat["opinions"].items(): st.write(f"**{tgt}** ➜ {txt.splitlines()[-1][:100]}…")
    if st.button("Show relation table", key="show_rel"):
        import pandas as pd
        allc=sorted(st.session_state.characters)
        df=pd.DataFrame("",index=allc,columns=allc)
        for cn in allc:
            op=load_json(CHAR_DIR/f"{cn}_thoughts.json",{"opinions":{}})["opinions"]
            for tg,tx in op.items(): df.loc[cn,tg]=tx.splitlines()[-1][:40]
        st.dataframe(df)

# -------------------- MAIN AREA (timeline + chat) -------------
st.title("🎭 Modular Role-Play")
st.subheader(f"Scene ▸ {st.session_state.active_scene or 'None'}")
for e in reversed(st.session_state.timeline[-50:]):
    timestr=e["time"][11:19] if "time" in e else ""
    st.markdown(f"- **[{timestr}] {e['type'].upper()}** — `{e['actor']}`: {e['content']}")

st.markdown("---")
for m in st.session_state.chat_log[-200:]:
    content=m["content"]
    if not st.session_state.show_thoughts:
        content="\n".join(l for l in content.splitlines() if not l.lower().startswith("thought:"))
    img=st.session_state.characters.get(m["role"],{}).get("image","")
    if img and Path(img).exists(): st.image(img,width=50)
    st.markdown(f"**{m['role']}**:\n{content}")

# -------------------- TTS STREAM (append lines) ---------------
def tts_append(line:str):
    with TTS_FILE.open("a",encoding="utf-8") as f: f.write(line+"\n")

# -------------------- DEV UPDATE / THOUGHT STORAGE ------------
def store_thought(cn,thought):
    p=CHAR_DIR/f"{cn}_thoughts.json"; d=load_json(p,{"opinions":{},"events":[]})
    if thought.lower().startswith("about "):
        tgt,rest=thought[6:].split(":",1); d["opinions"][tgt.strip()]=(d["opinions"].get(tgt.strip(),"")+"\n"+rest.strip())[-400:]
    else:
        d["events"].append({"time":datetime.datetime.now().isoformat(),"text":thought})
        d["events"]=d["events"][-60:]
    save_json(p,d)

def auto_dev(narr):
    for cn,c in st.session_state.characters.items():
        if not c.get("in_scene"): continue
        summ=chat_backend(c["model"],[
            {"role":"system","content":"Summarise impact on you in ≤30 words."},
            {"role":"user","content":f"Old:\n{c['develop']}\nScene:\n{narr}"}],0.3)
        c["develop"]=(c["develop"]+"\n"+summ).strip()[-800:]; save_json(CHAR_DIR/f"{cn}.json",c)

# -------------------- TURN LOGIC -------------------------------------------
def run_turn(msg):
    ts=datetime.datetime.now().isoformat(timespec="seconds")
    if msg.strip():
        st.session_state.chat_log.append({"role":st.session_state.player_name,"content":f"Speech: {msg}"})
        st.session_state.timeline.append({"type":"player","actor":st.session_state.player_name,"content":msg,"time":ts})
        tts_append(f"{st.session_state.player_name}: {msg}")

    gm_seq=[
        {"role":"system","content":w["gm_system"]},
        {"role":"system","content":w["prompt"]},
        {"role":"system","content":scene["prompt"]},
        {"role":"system","content":w["gm_notes"]},
        *st.session_state.gm_hist[-10:],{"role":"user","content":msg or "[continue]"}]
    narr=chat_backend(w["gm_model"],gm_seq)
    st.session_state.gm_hist.append({"role":"assistant","content":narr})
    st.session_state.chat_log.append({"role":"Narrator","content":narr})
    st.session_state.timeline.append({"type":"narration","actor":"Narrator","content":narr,"time":ts})
    tts_append(f"Narrator: {narr}")

    for cn,c in st.session_state.characters.items():
        if not c.get("in_scene") or cn==st.session_state.player_name: continue
        hist=st.session_state.char_hist.setdefault(cn,[])
        sys=f"You are {cn}. {c['personality']} Develop:\n{c['develop']}"
        reply=chat_backend(c["model"],[{"role":"system","content":sys},*hist[-10:],
                        {"role":"user","content":f"Narration:\n{narr}\nReply Thought:.. Speech:.."}])
        hist.append({"role":"assistant","content":reply})
        st.session_state.chat_log.append({"role":cn,"content":reply})
        st.session_state.timeline.append({"type":"character","actor":cn,"content":reply,"time":ts})
        for l in reply.splitlines():
            if l.lower().startswith("thought:"): store_thought(cn,l.split(":",1)[1].strip())
            if l.lower().startswith("speech:"): tts_append(f"{cn}: {l.split(':',1)[1].strip()}")

    auto_dev(narr)
    save_json(st.session_state.current_session,{"chat_log":st.session_state.chat_log,
              "timeline":st.session_state.timeline,"gm_hist":st.session_state.gm_hist,
              "char_hist":st.session_state.char_hist})

# -------------------- INPUT -----------------------------------------------
inp=st.chat_input("Your action / dialogue…", key="user_input")
if inp is not None: run_turn(inp); st.experimental_rerun()
