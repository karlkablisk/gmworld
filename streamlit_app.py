# rp_ai_chat.py - Memory compression, subjective/objective, debug logs, token tracking
import os, json, uuid, datetime, textwrap, base64, hashlib
from pathlib import Path
import requests
import streamlit as st

try:
    from gtts import gTTS
    tts_available = True
    tts_error_reported = False
except ImportError:
    tts_available = False
    tts_error_reported = False

try:
    import tiktoken  # For OpenAI token usage
    has_tiktoken = True
except ImportError:
    has_tiktoken = False

SAVE_ROOT = Path("rp_saves")
CHAR_DIR, SCENE_DIR, SESSION_DIR = (SAVE_ROOT / p for p in ("characters", "scenes", "sessions"))
WORLD_FILE = SAVE_ROOT / "world.json"
PERSONALITY_FILE = "personalityprompt.txt"
TTS_CACHE = SAVE_ROOT / "tts_cache"
MIND_CACHE = SAVE_ROOT / "mind_cache"

# First, create SAVE_ROOT
SAVE_ROOT.mkdir(exist_ok=True)
# Then create all subfolders
for p in (CHAR_DIR, SCENE_DIR, SESSION_DIR, TTS_CACHE, MIND_CACHE):
    p.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
A1111_URL = "http://localhost:7860"

for p in (CHAR_DIR, SCENE_DIR, SESSION_DIR):
    p.mkdir(parents=True, exist_ok=True)

def get_backend_models(backend):
    if backend == "ollama":
        try:
            tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=8).json()
            return ["gemma3:latest"] + sorted({m["name"] for m in tags.get("models", [])})
        except Exception:
            return ["gemma3:latest"]
    else:
        return ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-micro"]

st.sidebar.header("⚙️ Settings")
backend = st.sidebar.selectbox("AI backend", ["ollama", "openai"], key="backend_select")
MODELS_CACHE = get_backend_models(backend)

if backend == "openai":
    import openai
    openai.api_key = st.sidebar.text_input("OpenAI API key", type="password",
                                           value=st.session_state.get("openai_key",""),
                                           key="openai_key")
    st.session_state.openai_key = openai.api_key

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

def load_json(path: Path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default
def save_json(path: Path, obj): path.write_text(json.dumps(obj,indent=2), encoding="utf-8")

def count_tokens(text, model="gpt-4o"):
    if has_tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: estimate 1 token ≈ 4 chars
    return max(1, len(text)//4)

PERSONALITY_TEMPLATE = Path(PERSONALITY_FILE).read_text(encoding="utf-8") \
    if Path(PERSONALITY_FILE).exists() else "TEMPLATE NOT FOUND"

# -------------------- PARTY/SCENE SIDEBAR -----------------------
with st.sidebar.expander("🟢 Party Members", expanded=True):
    party_chars = [c for c in st.session_state.get("characters",{}).values() if c.get("party")]
    for char in party_chars:
        if st.button(f"{char['name']}", key=f"party_{char['name']}"):
            st.session_state.char_tab_selected = char["name"]

with st.sidebar.expander("🟡 Scene Members", expanded=True):
    scene_chars = []
    scene_name = st.session_state.get("active_scene")
    if scene_name and scene_name in st.session_state.get("scenes",{}):
        sc = st.session_state.scenes[scene_name]
        for cname in sc.get("chars", []):
            c = st.session_state.characters.get(cname)
            if c:
                scene_chars.append(c)
    for char in scene_chars:
        if st.button(f"{char['name']}", key=f"scene_{char['name']}"):
            st.session_state.char_tab_selected = char["name"]

# ====================== STATE INITIALIZATION ===================

if "init" not in st.session_state:
    st.session_state.init=True
    st.session_state.world = load_json(WORLD_FILE, {...})
    st.session_state.characters = ...
    st.session_state.scenes = ...
    st.session_state.current_session=None
    st.session_state.chat_log=[]; st.session_state.timeline=[]
    st.session_state.gm_hist=[];  st.session_state.char_hist={}
    st.session_state.active_scene=None
    st.session_state.player_name="PLAYER"
    st.session_state.show_thoughts=True
    st.session_state.char_tab_selected = None

# ENSURE char_tab_selected always exists for tab logic
if "char_tab_selected" not in st.session_state:
    st.session_state.char_tab_selected = None


# ================ CHARACTER/GM "MIND" MEMORY UTILS ================
def mind_path(name): return MIND_CACHE / f"{name}_mind.json"
def load_mind(name):
    d = load_json(mind_path(name), {
        "events":[], "thoughts":[], "opinions":{}, "knowledge":[],
        "develop":"", "said_log":[], "token_usage":0, "memory_mode":"subjective"
    })
    return d
def save_mind(name, d):
    save_json(mind_path(name), d)

def log_event(mind, text, ev_type="event", who=None):
    mind["events"].append({"time":datetime.datetime.now().isoformat(),
        "type":ev_type, "text":text, "who":who})
    mind["events"] = mind["events"][-60:]

def log_thought(mind, text):
    mind["thoughts"].append({"time":datetime.datetime.now().isoformat(),"text":text})
    mind["thoughts"] = mind["thoughts"][-60:]

def log_opinion(mind, about, text):
    if about not in mind["opinions"]: mind["opinions"][about] = []
    mind["opinions"][about].append({"time":datetime.datetime.now().isoformat(),"text":text})
    mind["opinions"][about] = mind["opinions"][about][-24:]

def log_knowledge(mind, text):
    mind["knowledge"].append({"time":datetime.datetime.now().isoformat(),"text":text})
    mind["knowledge"] = mind["knowledge"][-30:]

def log_said(mind, text):
    mind["said_log"].append({"time":datetime.datetime.now().isoformat(),"text":text})
    mind["said_log"] = mind["said_log"][-150:]

def summarize_entries(name, mind, N=10):
    for field in ("events","thoughts"):
        L = mind[field]
        if len(L) >= N:
            prompt = f"You are {'an impartial observer' if mind['memory_mode']=='objective' else name+'.'}\n"
            prompt += f"Summarize these {field} into one or two paragraphs in {mind['memory_mode']} memory style. If subjective, include bias, feelings, and personality:\n"
            for e in L[:N]:
                prompt += f"- {e['text']}\n"
            summary = chat_backend(st.session_state.world["gm_model"], [
                {"role":"system","content":prompt}
            ], temp=0.4)
            mind["develop"] += ("\n"+summary).strip()[-800:]
            del L[:N]
    for k,opinlist in mind["opinions"].items():
        if len(opinlist) >= N:
            prompt = f"Summarize these opinions about {k} into a single, in-character subjective statement:\n"
            prompt += "\n".join(f"- {o['text']}" for o in opinlist[:N])
            summary = chat_backend(st.session_state.world["gm_model"], [
                {"role":"system","content":prompt}
            ], temp=0.4)
            opinlist[:N] = [{"time":datetime.datetime.now().isoformat(), "text":summary}]
    # For knowledge, you can use same logic if desired.

def update_token_usage(name, mind, model):
    try:
        total_txt = "\n".join(
            [e['text'] for e in mind['events'][-10:]] +
            [t['text'] for t in mind['thoughts'][-10:]] +
            [o['text'] for k in mind['opinions'] for o in mind['opinions'][k][-6:]] +
            [k['text'] for k in mind['knowledge'][-8:]] +
            [mind.get("develop","")]
        )
        mind["token_usage"] = count_tokens(total_txt, model)
    except Exception:
        pass

# ================ TTS AUDIO UTILITY ================
def safe_tts(text, speaker):
    global tts_error_reported
    text = text.strip()
    if not text: return None
    fname = f"{speaker}_{hashlib.sha1(text.encode()).hexdigest()[:12]}.mp3"
    path = TTS_CACHE / fname
    if path.exists():
        return path
    if not tts_available:
        if not tts_error_reported:
            st.warning("TTS audio engine (gTTS) is not available.")
            tts_error_reported = True
        return None
    try:
        tts = gTTS(text)
        tts.save(str(path))
        return path
    except Exception as e:
        if not tts_error_reported:
            st.warning(f"TTS generation error (audio may not play): {e}")
            tts_error_reported = True
        return None


# =============== UI: Character Editor ================

tab_settings, tab_world, tab_scene, tab_char, tab_event, tab_thought = st.sidebar.tabs(
    ["⚙️ Settings", "🌍 World", "🎬 Scene", "👥 Characters", "📜 Events", "🧠 Thoughts"]
)

with tab_settings:
    st.checkbox("Show thoughts in chat", key="show_thoughts")

with tab_world:
    w = st.session_state.world
    w["prompt"]    = st.text_area("World prompt", w["prompt"], height=90, key="world_prompt")
    w["gm_system"] = st.text_area("GM system prompt", w["gm_system"], height=70, key="gm_sys")
    w["gm_notes"]  = st.text_area("GM notes / plot", w["gm_notes"], height=70, key="gm_notes")
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
    for cn,c in st.session_state.characters.items():
        c["in_scene"]=c.get("party") or cn in scene["chars"]

with tab_char:
    auto_select = st.session_state.get("char_tab_selected")
    chars = ["<new>"]+sorted(st.session_state.characters)
    if auto_select and auto_select in st.session_state.characters:
        sel = auto_select
        st.session_state.char_tab_selected = None
    else:
        sel = st.selectbox("Select", chars, key="char_select")
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

# ------------------ TTS UTILITY -------------------
def safe_tts(text, speaker):
    # returns path to .mp3 if successful, else None. Uses gTTS.
    text = text.strip()
    if not text: return None
    fname = f"{speaker}_{hashlib.sha1(text.encode()).hexdigest()[:12]}.mp3"
    path = TTS_CACHE / fname
    if path.exists():
        return path
    if not tts_available:
        global tts_error_reported
        if not tts_error_reported:
            st.warning("TTS audio engine (gTTS) is not available.")
            tts_error_reported = True
        return None
    try:
        tts = gTTS(text)
        tts.save(str(path))
        return path
    except Exception as e:
        global tts_error_reported
        if not tts_error_reported:
            st.warning(f"TTS generation error (audio may not play): {e}")
            tts_error_reported = True
        return None

# ========== MAIN CHAT/REPLY, MEMORY & TTS HANDLING ==========
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
    lines = content.splitlines()
    for line in lines:
        if m["role"].lower() == "narrator" or line.lower().startswith("speech:"):
            speaker = m["role"]
            spoken = line.split(":",1)[1].strip() if line.lower().startswith("speech:") else line.strip()
            mp3path = safe_tts(spoken, speaker)
            if mp3path:
                audio_bytes = mp3path.read_bytes()
                st.audio(audio_bytes, format="audio/mp3", start_time=0)
        st.markdown(f"**{m['role']}**: {line}")

def run_turn(msg):
    ts=datetime.datetime.now().isoformat(timespec="seconds")
    w = st.session_state.world
    scene_name = st.session_state.active_scene
    # (load GM mind)
    gm_mind = load_mind("GM")
    if msg.strip():
        st.session_state.chat_log.append({"role":st.session_state.player_name,"content":f"Speech: {msg}"})
        st.session_state.timeline.append({"type":"player","actor":st.session_state.player_name,"content":msg,"time":ts})
        # Debug log player utterances if you want (not necessary for now)
    # GM NARRATION
    gm_seq=[
        {"role":"system","content":w["gm_system"]},
        {"role":"system","content":w["prompt"]},
        {"role":"system","content":st.session_state.scenes.get(scene_name,{}).get("prompt","")},
        {"role":"system","content":w["gm_notes"]},
        *st.session_state.gm_hist[-10:],{"role":"user","content":msg or "[continue]"}]
    narr=chat_backend(w["gm_model"], gm_seq)
    st.session_state.gm_hist.append({"role":"assistant","content":narr})
    st.session_state.chat_log.append({"role":"Narrator","content":narr})
    st.session_state.timeline.append({"type":"narration","actor":"Narrator","content":narr,"time":ts})
    log_event(gm_mind, narr, ev_type="narration")
    summarize_entries("GM", gm_mind)
    update_token_usage("GM", gm_mind, w["gm_model"])
    save_mind("GM", gm_mind)

    for cn,c in st.session_state.characters.items():
        if not c.get("in_scene") or cn==st.session_state.player_name: continue
        mind = load_mind(cn)
        hist=st.session_state.char_hist.setdefault(cn,[])
        sys=f"You are {cn}. {c['personality']} Develop:\n{mind['develop']}"
        reply=chat_backend(c["model"],[{"role":"system","content":sys},*hist[-10:],
                        {"role":"user","content":f"Narration:\n{narr}\nReply Thought:.. Speech:.."}])
        hist.append({"role":"assistant","content":reply})
        st.session_state.chat_log.append({"role":cn,"content":reply})
        st.session_state.timeline.append({"type":"character","actor":cn,"content":reply,"time":ts})
        log_said(mind, reply)
        for l in reply.splitlines():
            if l.lower().startswith("thought:"):
                log_thought(mind, l.split(":",1)[1].strip())
            if l.lower().startswith("about "):
                about,rest = l[6:].split(":",1)
                log_opinion(mind, about.strip(), rest.strip())
            if l.lower().startswith("learned:"):
                log_knowledge(mind, l.split(":",1)[1].strip())
            log_event(mind, narr, ev_type="scene", who=cn)
        summarize_entries(cn, mind)
        update_token_usage(cn, mind, c["model"])
        save_mind(cn, mind)

    save_json(st.session_state.current_session,{"chat_log":st.session_state.chat_log,
              "timeline":st.session_state.timeline,"gm_hist":st.session_state.gm_hist,
              "char_hist":st.session_state.char_hist})

inp=st.chat_input("Your action / dialogue…", key="user_input")
if inp is not None: run_turn(inp); st.experimental_rerun()

# ------------- Optional TTS generator for scene/char images ----------------
def generate_image(prompt: str, save_path: str) -> bool:
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
