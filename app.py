import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import io
import json
import os
from datetime import datetime, timedelta
from scipy import stats as scipy_stats
from bs4 import BeautifulSoup
from supabase import create_client, Client

# ── SUPABASE CLIENT ───────────────────────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

st.set_page_config(page_title="AFL Fantasy DFS", page_icon="🏉", layout="wide")

# ── CONSTANTS ─────────────────────────────────────────────────

SCORING = {
    'kicks': 3, 'handballs': 2, 'marks': 3, 'tackles': 4,
    'goals': 6, 'behinds': 1, 'hit_outs': 1,
    'frees_for': 1, 'frees_against': -3,
}
STAT_COLS  = list(SCORING.keys())
POSITIONS  = ['MID', 'DEF', 'FWD', 'RUC']
SEASON_WEIGHTS = {2026: 1.00, 2025: 0.70, 2024: 0.40, 2023: 0.20}

DS_TEAM_MAP = {
    'ADE':'Adelaide','BRI':'Brisbane','CAR':'Carlton','COL':'Collingwood',
    'ESS':'Essendon','FRE':'Fremantle','GEE':'Geelong','GCS':'Gold Coast',
    'GWS':'GWS','HAW':'Hawthorn','MEL':'Melbourne','NOR':'North Melbourne',
    'POR':'Port Adelaide','RIC':'Richmond','STK':'St Kilda','SYD':'Sydney',
    'WCE':'West Coast','WBD':'Western Bulldogs',
    'Adelaide Crows':'Adelaide','Adelaide':'Adelaide',
    'Brisbane Lions':'Brisbane','Brisbane':'Brisbane',
    'Carlton':'Carlton','Collingwood':'Collingwood','Essendon':'Essendon',
    'Fremantle':'Fremantle','Geelong Cats':'Geelong','Geelong':'Geelong',
    'Gold Coast Suns':'Gold Coast','Gold Coast SUNS':'Gold Coast','Gold Coast':'Gold Coast',
    'GWS GIANTS':'GWS','GWS Giants':'GWS','Greater Western Sydney':'GWS',
    'Hawthorn':'Hawthorn','Melbourne':'Melbourne','North Melbourne':'North Melbourne',
    'Port Adelaide':'Port Adelaide','Port Adelaide Power':'Port Adelaide',
    'Richmond':'Richmond','St Kilda':'St Kilda',
    'Sydney Swans':'Sydney','Sydney':'Sydney',
    'West Coast Eagles':'West Coast','West Coast':'West Coast',
    'Western Bulldogs':'Western Bulldogs','Bulldogs':'Western Bulldogs',
}

DS_POSITION_MAP = {
    'MID':'MID','midfielder':'MID','Midfielder':'MID','Mid':'MID',
    'DEF':'DEF','defender':'DEF','Defender':'DEF','Def':'DEF',
    'FWD':'FWD','forward':'FWD','Forward':'FWD','Fwd':'FWD',
    'RUC':'RUC','ruck':'RUC','Ruck':'RUC','Ruc':'RUC','RK':'RUC',
    'MID/FWD':'MID','DEF/MID':'MID','FWD/MID':'MID','DEF/FWD':'DEF','MID/DEF':'MID',
}

FIXTURE_VENUE_MAP = {
    'MCG':'MCG','Marvel Stadium':'Marvel Stadium','SCG':'SCG','Gabba':'Gabba',
    'Adelaide Oval':'Adelaide Oval','Optus Stadium':'Optus Stadium',
    'GMHBA Stadium':'GMHBA Stadium','ENGIE Stadium':'Giants Stadium',
    'GIANTS Stadium':'Giants Stadium','People First Stadium':'Carrara',
    'Heritage Bank Stadium':'Carrara','Ninja Stadium':'Blundstone Arena',
    'Blundstone Arena':'Blundstone Arena','Barossa Park':'Barossa Park',
    'Corroboree Group Oval Manuka':'Manuka Oval','Manuka Oval':'Manuka Oval',
    'TIO Stadium':'TIO Stadium','TIO Traeger Park':'Traeger Park',
    'UTAS Stadium':'University of Tasmania Stadium',
    'University of Tasmania Stadium':'University of Tasmania Stadium',
    'Mars Stadium':'Mars Stadium','Norwood Oval':'Norwood Oval',
    'Hands Oval':'Hands Oval','Adelaide Hills':'Mount Barker',
    'Cazalys Stadium':'Cazalys Stadium',
}

AFLT_VENUE_MAP = {
    'M.C.G.':'MCG','Docklands':'Marvel Stadium','S.C.G.':'SCG','Gabba':'Gabba',
    'Adelaide Oval':'Adelaide Oval','Perth Stadium':'Optus Stadium',
    'Kardinia Park':'GMHBA Stadium','Sydney Showground':'Giants Stadium',
    'Carrara':'Carrara','York Park':'University of Tasmania Stadium',
    'Bellerive Oval':'Blundstone Arena','Marrara Oval':'TIO Stadium',
    'Traeger Park':'Traeger Park','Barossa Oval':'Barossa Park',
    'Barossa Park':'Barossa Park','Manuka Oval':'Manuka Oval',
    'Norwood Oval':'Norwood Oval','Hands Oval':'Hands Oval',
    'Mars Stadium':'Mars Stadium','Eureka Stadium':'Mars Stadium',
    'Subiaco':'Subiaco Oval','Mount Barker':'Mount Barker',
    'Cazalys Stadium':'Cazalys Stadium',
}

VENUE_CITY = {
    'MCG':                            ('Melbourne',     -37.8200,  144.9830),
    'Marvel Stadium':                 ('Melbourne',     -37.8167,  144.9472),
    'SCG':                            ('Sydney',        -33.8915,  151.2246),
    'Gabba':                          ('Brisbane',      -27.4858,  153.0381),
    'Adelaide Oval':                  ('Adelaide',      -34.9158,  138.5960),
    'Optus Stadium':                  ('Perth',         -31.9505,  115.8890),
    'GMHBA Stadium':                  ('Geelong',       -38.1574,  144.3550),
    'Giants Stadium':                 ('Sydney',        -33.8473,  150.9905),
    'Carrara':                        ('Gold Coast',    -27.9292,  153.3686),
    'Blundstone Arena':               ('Hobart',        -42.8794,  147.3294),
    'Barossa Park':                   ('Adelaide',      -34.5667,  138.8833),
    'Manuka Oval':                    ('Canberra',      -35.3200,  149.1300),
    'TIO Stadium':                    ('Darwin',        -12.4634,  130.8456),
    'Traeger Park':                   ('Alice Springs', -23.6980,  133.8807),
    'University of Tasmania Stadium': ('Launceston',    -41.4545,  147.1358),
    'Mars Stadium':                   ('Ballarat',      -37.5500,  143.8500),
    'Norwood Oval':                   ('Adelaide',      -34.9158,  138.5960),
    'Hands Oval':                     ('Bunbury',       -33.3271,  115.6414),
    'Mount Barker':                   ('Adelaide',      -35.0700,  138.8600),
    'Subiaco Oval':                   ('Perth',         -31.9505,  115.8890),
    'Cazalys Stadium':                ('Cairns',        -16.9186,  145.7781),
}

# CHANGE 1: Added Will Hayes and Will Edwards
NAME_CORRECTIONS = {
    'Tom Lynch':'Tom_Lynch1','Bailey Williams':'Bailey_Williams0',
    'Bailey J. Williams':'Bailey_Williams1','Matthew Kennedy':'Matthew_Kennedy1',
    'Harrison Petty':'Harry_Petty','Jamie Elliott':'Jamie_Elliott1',
    'Jack Henry':'Jack_Henry1','Harrison Jones':'Harry_Jones2',
    'Maurice Rioli':'Maurice_Rioli1','Arthur Jones':'Arthur_Jones1',
    'Oscar Steene':'Oscar_Steene','Robert Hansen Jr':'Robert_Hansen',
    'Matthew Carroll':'Matt_Carroll','Archie Roberts':'Archie_Roberts1',
    'Archer May':'Archie_May',"Reilly O'Brien":'Reilly_OBrien',
    "Mark O'Connor":'Mark_OConnor',"Jaeger O'Meara":'Jaeger_OMeara',
    "Nathan O'Driscoll":'Nathan_ODriscoll',"James O'Donnell":'James_ODonnell',
    "Connor O'Sullivan":'Connor_OSullivan',"Harry O'Farrell":'Harry_OFarrell',
    "Finn O'Sullivan":'Finn_OSullivan',"Balyn O'Brien":'Balyn_OBrien',
    'Jordan De Goey':'Jordan_de_Goey','Christopher Scerri':'Chris_Scerri',
    'Leonardo Lombard':'Leo_Lombard','Lachie Jaques':'Lachie_Jaques',
    'Luke Trainor':'Luke_Trainor1','Sam Butler':'Sam_Butler1',
    'Matt Duffy':'Matthew_Duffy','Matt Hill':'Matthew_Hill',
    'Tom Campbell':'Tom_Campbell1','Mitchell Hinge':'Mitch_Hinge',
    'Callum Brown':'Callum_Brown1','Jack Buckley':'Jack_Buckley1',
    'Tom Green':'Tom_Green1','Henry Smith':'Henry_Smith1',
    'Jack Carroll':'Jack_Carroll1','Joshua Draper':'Josh_Draper',
    'Nicholas Madden':'Nick_Madden','Thomas Edwards':'Tom_Edwards',
    'William Hayes':'Will_Hayes1','William Edwards':'Will_Edwards',
}

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

DISPOSAL_LINES = list(range(10, 45, 5))
KICK_LINES     = list(range(4,  28, 2))
HANDBALL_LINES = list(range(4,  28, 2))
MARK_LINES     = list(range(2,  14, 2))
TACKLE_LINES   = list(range(2,  14, 2))
HITOUT_LINES   = list(range(10, 45, 5))

FACTOR_KEYS   = ['form', 'trend', 'opponent', 'venue', 'home_away', 'weather', 'tog']
FACTOR_LABELS = ['Form (last 5)', 'Trend (20-game)', 'Opponent difficulty',
                 'Venue history', 'Home/Away', 'Weather', 'TOG']

# ── HELPERS ───────────────────────────────────────────────────

def winsorise(vals, lower=10, upper=90):
    if len(vals) < 4: return vals
    return np.clip(vals, np.percentile(vals, lower), np.percentile(vals, upper))

def wavg(series):
    vals = winsorise(np.array(series, dtype=float))
    if not len(vals): return np.nan
    w = np.exp(np.linspace(-1, 0, len(vals)))
    return float(np.dot(vals, w / w.sum()))

def calc_trend(series):
    vals = winsorise(np.array(series, dtype=float))
    if len(vals) < 5: return 1.0
    sl, _, _, p, _ = scipy_stats.linregress(np.arange(len(vals)), vals)
    m = np.mean(vals)
    if m == 0 or p > 0.15: return 1.0
    return float(1.0 + np.clip(sl / m * len(vals) * 0.5, -0.10, 0.10))

def calc_over_prob(proj, std, line):
    if std <= 0: return 1.0 if proj > line else 0.0
    return round(float(1 - scipy_stats.norm.cdf((line + 0.5 - proj) / std)), 3)

def build_afltables_url(player_name):
    if player_name in NAME_CORRECTIONS:
        slug = NAME_CORRECTIONS[player_name]
        return f'https://afltables.com/afl/stats/players/{slug[0].upper()}/{slug}.html'
    name_fixed = player_name.strip().replace("'", "")
    name_url   = name_fixed.replace(' ', '_')
    return f'https://afltables.com/afl/stats/players/{name_fixed[0].upper()}/{name_url}.html'

# ── DATA LOADING ──────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def load_stats():
    sb = get_supabase()
    all_rows = []
    chunk = 1000
    offset = 0
    while True:
        resp = sb.table('player_stats').select('*').range(offset, offset+chunk-1).execute()
        if not resp.data: break
        all_rows.extend(resp.data)
        if len(resp.data) < chunk: break
        offset += chunk
    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    for col in STAT_COLS + ['fantasy_score', 'tog_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

@st.cache_data(show_spinner=False, ttl=300)
def load_roster():
    sb = get_supabase()
    resp = sb.table('player_roster').select('*').execute()
    if not resp.data: return pd.DataFrame()
    return pd.DataFrame(resp.data)

def save_stats_to_supabase(new_records):
    sb = get_supabase()
    if not new_records: return
    sb.table('player_stats').upsert(new_records, on_conflict='name,season,round,opponent').execute()
    st.cache_data.clear()

def save_roster_to_supabase(df_roster):
    sb = get_supabase()
    records = df_roster.to_dict('records')
    sb.table('player_roster').upsert(records, on_conflict='player_id').execute()
    st.cache_data.clear()

def load_saved_slates():
    sb = get_supabase()
    resp = sb.table('saved_slates').select('*').execute()
    if not resp.data: return {}
    result = {}
    for row in resp.data:
        try:
            result[row['name']] = json.loads(row['data']) if isinstance(row['data'], str) else row['data']
        except: pass
    return result

def save_slate_to_supabase(name, data):
    sb = get_supabase()
    serialisable = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            serialisable[k] = v.to_dict('records')
        elif isinstance(v, set):
            serialisable[k] = list(v)
        else:
            serialisable[k] = v
    sb.table('saved_slates').upsert(
        {'name': name, 'data': json.dumps(serialisable)}, on_conflict='name'
    ).execute()

def load_factor_weights():
    sb = get_supabase()
    try:
        resp = sb.table('factor_weights').select('*').eq('id', 1).execute()
        if resp.data:
            w = resp.data[0]['weights']
            return json.loads(w) if isinstance(w, str) else w
    except:
        pass
    return {}

def save_factor_weights(weights):
    sb = get_supabase()
    sb.table('factor_weights').upsert(
        {'id': 1, 'weights': json.dumps(weights)}
    ).execute()

def load_app_prefs():
    sb = get_supabase()
    try:
        resp = sb.table('factor_weights').select('*').eq('id', 2).execute()
        if resp.data:
            p = resp.data[0]['weights']
            return json.loads(p) if isinstance(p, str) else p
    except:
        pass
    return {}

def save_app_prefs(prefs):
    sb = get_supabase()
    try:
        sb.table('factor_weights').upsert(
            {'id': 2, 'weights': json.dumps(prefs)}
        ).execute()
    except:
        pass

# ── WEATHER ───────────────────────────────────────────────────

def fetch_weather(lat, lon):
    try:
        url  = (f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
                f'&daily=weathercode,precipitation_sum,windspeed_10m_max'
                f'&timezone=Australia%2FSydney&forecast_days=7')
        d    = requests.get(url, timeout=8).json()
        rain = max(d['daily']['precipitation_sum'][:3])
        wind = max(d['daily']['windspeed_10m_max'][:3])
        if rain > 8:  return 'heavy_rain', rain, wind
        if rain > 2:  return 'light_rain',  rain, wind
        if wind > 40: return 'wind',         rain, wind
        return 'fine', rain, wind
    except:
        return 'fine', 0, 0

def fetch_all_venue_weather(fixtures_list):
    wm = {}
    for f in fixtures_list:
        v = f.get('venue', 'TBC')
        if v in wm or v == 'TBC': continue
        if v in VENUE_CITY:
            _, lat, lon  = VENUE_CITY[v]
            cond, _, _   = fetch_weather(lat, lon)
            wm[v] = cond
        else:
            wm[v] = 'fine'
    return wm

# ── DRAFTSTARS CSV PARSER ─────────────────────────────────────

def parse_draftstars_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.lower().strip() for c in df.columns]

    name_col   = next((c for c in df.columns if c == 'name' or ('name' in c and 'nick' not in c)), None)
    team_col   = next((c for c in df.columns if 'team' in c), None)
    pos_col    = next((c for c in df.columns if 'position' in c or c == 'pos'), None)
    game_col   = next((c for c in df.columns if 'game' in c or 'match' in c or 'fixture' in c), None)
    salary_col = next((c for c in df.columns if 'salary' in c or 'price' in c), None)
    status_col = next((c for c in df.columns if 'status' in c or 'playing' in c), None)

    if not all([name_col, team_col, pos_col]):
        raise ValueError(f'Missing required columns. Found: {list(df.columns)}')

    df['_pos']  = df[pos_col].map(DS_POSITION_MAP).fillna(df[pos_col].str.upper().str.strip())
    df['_team'] = df[team_col].map(DS_TEAM_MAP).fillna(df[team_col].str.strip())

    def combine_pos(s):
        order = {'DEF':0,'MID':1,'RUC':2,'FWD':3}
        u = list(dict.fromkeys(p for p in s if pd.notna(p) and p != ''))
        u.sort(key=lambda p: order.get(p, 9))
        return '/'.join(u) if u else 'MID'

    agg = {name_col:'first', '_team':'first', '_pos':combine_pos}
    if salary_col: agg[salary_col] = 'max'
    if game_col:   agg[game_col]   = 'first'

    players = df.groupby(name_col, as_index=False).agg(agg)
    rm = {name_col:'ds_name', '_team':'team', '_pos':'position'}
    if salary_col: rm[salary_col] = 'salary'
    if game_col:   rm[game_col]   = 'game_info'
    players = players.rename(columns=rm)
    players['player_id'] = players['ds_name']

    out_players = []
    if status_col and status_col in df.columns:
        active = df[
            df[status_col].str.upper().str.strip().str.contains('NAMED IN TEAM TO PLAY|CONFIRMED IN TEAM TO PLAY', na=False)
        ][name_col].unique().tolist()

        out_df = df[df[status_col].str.upper().str.strip() == 'OUT']
        seen = set()
        for _, row in out_df.iterrows():
            pname = row[name_col]
            if pname in seen: continue
            seen.add(pname)
            out_players.append({
                'name':     pname,
                'team':     DS_TEAM_MAP.get(str(row[team_col]).strip(), str(row[team_col]).strip()),
                'position': DS_POSITION_MAP.get(str(row[pos_col]).strip(), 'MID').split('/')[0],
            })
        players = players[players['ds_name'].isin(active)].reset_index(drop=True)

    players = players.dropna(subset=['ds_name']).reset_index(drop=True)
    return players, out_players

# ── SCRAPING ──────────────────────────────────────────────────

def scrape_player_afltables(player_name, team, position, seasons, venue_lookup, is_home_lookup=None):
    FINALS_LABELS = {'EF','QF','SF','PF','GF'}
    url = build_afltables_url(player_name)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200: return [], f'http_{resp.status_code}'
        soup    = BeautifulSoup(resp.text, 'lxml')
        tables  = soup.find_all('table')
        records = []
        target_seasons = set(str(s) for s in seasons)
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 3: continue
            sm = re.match(r'^(.+?)\s*-\s*(\d{4})$', rows[0].get_text(strip=True))
            if not sm or sm.group(2) not in target_seasons: continue
            season_int = int(sm.group(2))
            for row in rows[2:]:
                cells = [td.get_text(strip=True) for td in row.find_all(['td','th'])]
                if not cells or len(cells) < 10: continue
                first = cells[0].strip()
                if not first or 'Total' in first or 'Average' in first: continue
                try: int(re.sub(r'[\u2191\u2193\s]', '', first))
                except ValueError: continue
                opp_raw  = cells[1].strip() if len(cells) > 1 else ''
                opponent = DS_TEAM_MAP.get(opp_raw, opp_raw)
                rd_raw   = cells[2].strip() if len(cells) > 2 else ''
                if rd_raw in FINALS_LABELS:
                    round_num = rd_raw
                else:
                    try: round_num = int(re.search(r'\d+', rd_raw).group())
                    except: continue
                game_date = None
                for c in cells:
                    dm = re.search(r'(\d{1,2}-[A-Za-z]+-\d{4})', c)
                    if dm:
                        try: game_date = datetime.strptime(dm.group(1), '%d-%b-%Y').date()
                        except: pass
                        break
                venue = 'Unknown'; is_home = False
                if game_date:
                    t = DS_TEAM_MAP.get(team, team)
                    venue = (venue_lookup.get((t, opponent, game_date)) or
                             venue_lookup.get((opponent, t, game_date)) or 'Unknown')
                    if is_home_lookup:
                        is_home = (is_home_lookup.get((t, opponent, game_date)) or
                                   is_home_lookup.get((opponent, t, game_date)) or False)
                def gi(i):
                    if i >= len(cells) or not cells[i].strip(): return 0
                    try: return max(0, int(float(cells[i].strip())))
                    except: return 0
                def gt(i, d=0.85):
                    if i >= len(cells) or not cells[i].strip(): return d
                    try:
                        v = float(cells[i].strip().replace('%',''))
                        return round(v/100.0 if v > 1 else v, 3)
                    except: return d
                kicks=gi(5); marks=gi(6); handballs=gi(7); goals=gi(9)
                behinds=gi(10); hit_outs=gi(11); tackles=gi(12)
                frees_for=gi(17); frees_against=gi(18); tog=gt(27)
                fs = (kicks*3 + handballs*2 + marks*3 + tackles*4 + goals*6
                      + behinds + hit_outs + frees_for + frees_against*-3)
                records.append({
                    'name':player_name,'team':team,'position':position.split('/')[0],
                    'position_full':position,'season':season_int,'round':round_num,
                    'opponent':opponent,'venue':venue,'is_home':is_home,'tog_pct':tog,
                    'kicks':kicks,'handballs':handballs,'marks':marks,'tackles':tackles,
                    'goals':goals,'behinds':behinds,'hit_outs':hit_outs,
                    'frees_for':frees_for,'frees_against':frees_against,'fantasy_score':fs,
                    'game_date':str(game_date) if game_date else '',
                })
        return records, 'ok'
    except Exception as e:
        return [], f'error:{e}'

# CHANGE 3: Added middle initial removal and suffix stripping fallbacks
def scrape_with_fallbacks(player_name, team, position, seasons, venue_lookup, is_home_lookup=None):
    EXPANSIONS = {
        'Matt':'Matthew','Tom':'Thomas','Will':'William','Sam':'Samuel',
        'Ben':'Benjamin','Dan':'Daniel','Lachie':'Lachlan','Josh':'Joshua',
        'Jake':'Jacob','Alex':'Alexander','Nick':'Nicholas','Mitch':'Mitchell',
        'Charlie':'Charles','Pat':'Patrick','Cam':'Cameron','Zac':'Zachary',
        'Zak':'Zachary','Rob':'Robert','Mike':'Michael','Fred':'Frederick',
        'Ollie':'Oliver','Archie':'Archibald','Harry':'Harrison','Jack':'Jackson',
    }
    r, s = scrape_player_afltables(player_name, team, position, seasons, venue_lookup, is_home_lookup)
    if s == 'ok' and r: return r, s

    # Try expanding short first name
    parts = player_name.split()
    if parts and parts[0] in EXPANSIONS:
        exp = EXPANSIONS[parts[0]] + ' ' + ' '.join(parts[1:])
        r, s = scrape_player_afltables(exp, team, position, seasons, venue_lookup, is_home_lookup)
        if s == 'ok' and r: return r, s

    # Try removing middle initials (e.g. "Tom J. Lynch" -> "Tom Lynch")
    ni = re.sub(r'\s+[A-Z]\.\s+', ' ', player_name).strip()
    if ni != player_name:
        r, s = scrape_player_afltables(ni, team, position, seasons, venue_lookup, is_home_lookup)
        if s == 'ok' and r: return r, s

    # Try removing suffixes (Jr., Sr.)
    ns = re.sub(r'\s+Jr\.?$|\s+Sr\.?$', '', player_name, flags=re.I).strip()
    if ns != player_name:
        r, s = scrape_player_afltables(ns, team, position, seasons, venue_lookup, is_home_lookup)
        if s == 'ok' and r: return r, s

    return [], 'not_found'

# ── PROJECTION MODEL ──────────────────────────────────────────

class AFLFantasyProjector:
    def __init__(self, df):
        self.df = df.copy()
        self._preprocess()
        self._build_opponent_ratings()
        self._build_venue_ratings()

    def _preprocess(self):
        FINALS_ORDER = {'EF':100,'QF':101,'SF':102,'PF':103,'GF':104}
        def rsort(r):
            r = str(r).strip()
            if r in FINALS_ORDER: return FINALS_ORDER[r]
            try: return int(r)
            except: return 999
        self.df['_rs'] = self.df['round'].map(rsort)
        self.df = self.df.sort_values(['name','season','_rs']).drop(columns='_rs').reset_index(drop=True)
        for col in STAT_COLS + ['fantasy_score','tog_pct']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def _build_opponent_ratings(self):
        self.opp_ratings = {}
        for pos in POSITIONS:
            ws, wt = {}, {}
            for season, w in SEASON_WEIGHTS.items():
                s = self.df[(self.df['season']==season) & (self.df['position']==pos)]
                if not len(s): continue
                for opp, val in s.groupby('opponent')['fantasy_score'].mean().items():
                    ws[opp] = ws.get(opp,0) + val*w
                    wt[opp] = wt.get(opp,0) + w
            bw  = {o: ws[o]/wt[o] for o in ws}
            avg = sum(bw.values())/len(bw) if bw else 1
            self.opp_ratings[pos] = {o: v/avg for o,v in bw.items()} if avg>0 else {}

    def _build_venue_ratings(self):
        recent = self.df[self.df['season'].isin(sorted(self.df['season'].unique())[-2:])]
        po = recent.groupby('name')['fantasy_score'].mean()
        self.venue_ratings = {}
        if 'venue' not in recent.columns: return
        for (p,v), s in recent.groupby(['name','venue'])['fantasy_score'].mean().items():
            ov = po.get(p, s)
            if ov > 0: self.venue_ratings[(p,v)] = s/ov

    def project_player(self, player_name, opponent, venue, is_home,
                       weather='fine', injury_override=None, tog_override=None,
                       factor_weights=None, position=None, team=None,
                       manual_base_scores=None):
        fw = factor_weights or {}
        manual_base_scores = manual_base_scores or {}
        FINALS_ORDER = {'EF':100,'QF':101,'SF':102,'PF':103,'GF':104}
        def rsort(r):
            r=str(r).strip()
            if r in FINALS_ORDER: return FINALS_ORDER[r]
            try: return int(r)
            except: return 999

        pd_ = self.df[self.df['name']==player_name].copy()
        pd_['_rs'] = pd_['round'].map(rsort)
        pd_ = pd_.sort_values(['season','_rs']).drop(columns='_rs')

        if not len(pd_):
            base = manual_base_scores.get(player_name)
            if base is None: return None
            pos = position or 'MID'
            of  = self.opp_ratings.get(pos, {}).get(opponent, 1.0)
            wmap = {'fine':1.00,'light_rain':0.97,'heavy_rain':0.92,'wind':0.95}
            wf   = wmap.get(weather, 1.00)
            proj = base * of * wf
            std  = proj * 0.40
            return {
                'player':player_name,'team':team or '','position':pos,
                'opponent':opponent,'venue':venue,'is_home':is_home,'weather':weather,
                'projection':round(proj,1),'median':round(proj,1),
                'floor':round(max(0,proj-1.5*std),1),'ceiling':round(proj+1.5*std,1),
                'confidence':30.0,'variance':40.0,'base_avg':round(base,1),
                'form_5_avg':None,'form_3_avg':None,
                'form_factor':1.0,'trend_factor':1.0,'opp_factor':round(of,3),
                'venue_factor':1.0,'home_away_factor':1.0,'weather_factor':round(wf,3),
                'tog_factor':1.0,'expected_tog':85.0,
            }

        pd_ = pd_[pd_['tog_pct']>=0.45].copy()
        if not len(pd_): return None

        pos  = pd_['position'].iloc[-1]
        team = pd_['team'].iloc[-1]
        r20  = pd_['fantasy_score'].tail(20).values
        r5   = pd_['fantasy_score'].tail(5).values
        r3   = pd_['fantasy_score'].tail(3).values

        base   = wavg(r20)
        median = round(float(np.median(winsorise(r20))), 1)
        ff     = float(np.clip(np.mean(r5)/base, 0.80, 1.20)) if len(r5)>=3 and base>0 else 1.0
        tf     = calc_trend(r20)
        of     = self.opp_ratings.get(pos,{}).get(opponent, 1.0)

        vf = 1.0
        if 'venue' in pd_.columns and len(pd_[pd_['venue']==venue])>=5:
            vf = float(np.clip(self.venue_ratings.get((player_name,venue),1.0), 0.93, 1.07))

        hf = 1.0
        if 'is_home' in pd_.columns and len(pd_)>=10:
            ha  = pd_[pd_['is_home']==True]['fantasy_score'].mean()
            aa  = pd_[pd_['is_home']==False]['fantasy_score'].mean()
            avg = pd_['fantasy_score'].mean()
            if avg>0:
                ref = ha if is_home else aa
                hf  = float(np.clip(ref/avg, 0.85, 1.15)) if ref>0 else 1.0

        wmap = {'fine':1.00,'light_rain':0.97,'heavy_rain':0.92,'wind':0.95}
        wf   = wmap.get(weather,1.00)
        if pos=='MID' and weather!='fine': wf = 1-(1-wf)*0.5

        exp_tog = float(tog_override) if tog_override else float(pd_['tog_pct'].tail(5).mean() or 0.85)
        avg_tog = float(pd_['tog_pct'].mean())
        tgf     = float(np.clip(exp_tog/avg_tog, 0.70, 1.00)) if avg_tog>0 else 1.0

        inj = float(injury_override) if injury_override else 1.0

        adj_ff  = 1.0+(ff -1.0)*fw.get('form',     1.0)
        adj_tf  = 1.0+(tf -1.0)*fw.get('trend',    1.0)
        adj_of  = 1.0+(of -1.0)*fw.get('opponent', 1.0)
        adj_vf  = 1.0+(vf -1.0)*fw.get('venue',    1.0)
        adj_hf  = 1.0+(hf -1.0)*fw.get('home_away',1.0)
        adj_wf  = 1.0+(wf -1.0)*fw.get('weather',  1.0)
        adj_tgf = 1.0+(tgf-1.0)*fw.get('tog',      1.0)

        proj = (base*(0.60+0.40*adj_ff)*(0.80+0.20*adj_tf)
                *adj_of*adj_vf*adj_hf*adj_wf*adj_tgf*inj)

        std = float(pd_['fantasy_score'].tail(10).std() or proj*0.25)
        cv  = std/proj if proj>0 else 1

        return {
            'player':player_name,'team':team,'position':pos,
            'opponent':opponent,'venue':venue,'is_home':is_home,'weather':weather,
            'projection':round(proj,1),'median':median,
            'floor':round(max(0,proj-1.5*std),1),'ceiling':round(proj+1.5*std,1),
            'confidence':round(float(np.clip(1-cv*0.5,0.3,0.95))*100,1),
            'variance':round(float(np.clip(cv*100,5,60)),1),
            'base_avg':round(base,1),
            'form_5_avg':round(float(np.mean(r5)),1) if len(r5) else None,
            'form_3_avg':round(float(np.mean(r3)),1) if len(r3) else None,
            'form_factor':round(ff,3),'trend_factor':round(tf,3),
            'opp_factor':round(of,3),'venue_factor':round(vf,3),
            'home_away_factor':round(hf,3),'weather_factor':round(wf,3),
            'tog_factor':round(tgf,3),'expected_tog':round(exp_tog*100,1),
        }

    def project_stat(self, player_name, opponent, is_home, weather, injury_override,
                     tog_override, factor_weights, opp_stat_ratings):
        fw  = factor_weights or {}
        pd_ = self.df[self.df['name']==player_name].copy().sort_values(['season','round'])
        if not len(pd_): return None
        pd_ = pd_[pd_['tog_pct']>=0.45].copy()
        if not len(pd_): return None
        position = pd_['position'].iloc[-1]
        r20 = pd_.tail(20); r5 = pd_.tail(5); r3 = pd_.tail(3)

        exp_tog  = float(tog_override) if tog_override else float(pd_['tog_pct'].tail(5).mean() or 0.85)
        avg_tog  = float(pd_['tog_pct'].mean())
        tog_f    = float(np.clip(exp_tog/avg_tog, 0.70, 1.00)) if avg_tog>0 else 1.0
        inj_f    = float(injury_override) if injury_override else 1.0
        kick_avg = wavg(r20['kicks'].values)
        hb_avg   = wavg(r20['handballs'].values)

        results = {}
        for stat, lines in [
            ('kicks',KICK_LINES),('handballs',HANDBALL_LINES),
            ('marks',MARK_LINES),('tackles',TACKLE_LINES),('hit_outs',HITOUT_LINES),
        ]:
            base    = wavg(r20[stat].values)
            med_raw = float(np.median(winsorise(r20[stat].values)))
            avg5    = float(r5[stat].mean()); avg3 = float(r3[stat].mean())
            form    = float(np.clip(avg5/base,0.80,1.20)) if base>0 else 1.0
            tr      = calc_trend(r20[stat].values)
            opp_f   = opp_stat_ratings.get(stat,{}).get(position,{}).get(opponent,1.0)

            total  = kick_avg+hb_avg
            kr     = kick_avg/total if total>0 else 0.5
            base_p = {'fine':0,'light_rain':0.04,'heavy_rain':0.10,'wind':0.06}.get(weather,0)
            if stat=='kicks':      wf = round(1-base_p*(0.5+kr),4)
            elif stat=='handballs':wf = round(1-max(base_p*(0.5-kr*0.5),-0.03),4)
            elif stat=='marks':    wf = {'fine':1.00,'light_rain':0.96,'heavy_rain':0.90,'wind':0.94}.get(weather,1.0)
            elif stat=='tackles':  wf = {'fine':1.00,'light_rain':1.02,'heavy_rain':1.04,'wind':1.01}.get(weather,1.0)
            else:                  wf = 1.0

            adj_opp  = 1.0+(opp_f-1.0)*fw.get('opponent',1.0)
            adj_wf   = 1.0+(wf   -1.0)*fw.get('weather', 1.0)
            adj_tog  = 1.0+(tog_f-1.0)*fw.get('tog',     1.0)
            adj_form = 0.60+0.40*(1.0+(form-1.0)*fw.get('form', 1.0))
            adj_tr   = 0.80+0.20*(1.0+(tr  -1.0)*fw.get('trend',1.0))

            proj_mean = round(max(0, base   *adj_form*adj_tr*adj_opp*adj_wf*adj_tog*inj_f),1)
            proj_med  = round(max(0, med_raw*adj_form*adj_tr*adj_opp*adj_wf*adj_tog*inj_f),1)
            std = float(pd_[stat].tail(10).std() or proj_med*0.30)
            results[stat] = {
                'proj':proj_med,'mean':proj_mean,
                'floor':round(max(0,proj_med-1.5*std),1),
                'ceiling':round(proj_med+1.5*std,1),
                'avg_20':round(base,1),'avg_5':round(avg5,1),'avg_3':round(avg3,1),'std':std,
                'ou':{f'{stat}_over_{l}':calc_over_prob(proj_med,std,l) for l in lines},
            }

        k = results['kicks']; h = results['handballs']
        ds     = round(float(np.sqrt(k['std']**2+h['std']**2)),2)
        dp_med = round(k['proj']+h['proj'],1)
        results['disposals'] = {
            'proj':dp_med,'mean':round(k['mean']+h['mean'],1),
            'floor':round(k['floor']+h['floor'],1),'ceiling':round(k['ceiling']+h['ceiling'],1),
            'avg_20':round(k['avg_20']+h['avg_20'],1),'avg_5':round(k['avg_5']+h['avg_5'],1),
            'avg_3':round(k['avg_3']+h['avg_3'],1),'std':ds,
            'ou':{f'disposals_over_{l}':calc_over_prob(dp_med,ds,l) for l in DISPOSAL_LINES},
        }
        return {'player':player_name,'position':position,**results}


def build_opp_stat_ratings(df_stats):
    ratings = {}
    for stat in ['kicks','handballs','marks','tackles','hit_outs']:
        ratings[stat] = {}
        for pos in POSITIONS:
            ws,wt = {},{}
            for season,w in SEASON_WEIGHTS.items():
                s = df_stats[(df_stats['season']==season)&(df_stats['position']==pos)]
                if not len(s): continue
                for opp,val in s.groupby('opponent')[stat].mean().items():
                    ws[opp]=ws.get(opp,0)+val*w; wt[opp]=wt.get(opp,0)+w
            bw  = {o:ws[o]/wt[o] for o in ws}
            avg = np.mean(list(bw.values())) if bw else 1
            ratings[stat][pos] = {o:v/avg for o,v in bw.items()} if avg>0 else {}
    return ratings


def run_projections(df_stats, ds_players, fixtures, weather_map,
                    injury_map, tog_map, factor_weights,
                    manual_base_scores, role_factors):
    if df_stats.empty or ds_players is None or not fixtures:
        return pd.DataFrame(), pd.DataFrame()

    projector = AFLFantasyProjector(df_stats)
    team_fix  = {}
    for f in fixtures:
        team_fix[f['home_team']] = {'opponent':f['away_team'],'venue':f['venue'],'is_home':True}
        team_fix[f['away_team']] = {'opponent':f['home_team'],'venue':f['venue'],'is_home':False}

    rows = []
    all_players = set(df_stats['name'].unique()) | set(manual_base_scores.keys())
    for p in all_players:
        pd_p = df_stats[df_stats['name']==p]
        pt   = pd_p['team'].iloc[-1] if len(pd_p) else None
        if pt is None:
            match = ds_players[ds_players['ds_name']==p]
            if len(match): pt = match['team'].iloc[0]
        if pt not in team_fix: continue
        f     = team_fix[pt]
        match = ds_players[ds_players['ds_name']==p]
        pos   = match['position'].iloc[0].split('/')[0] if len(match) else None
        team  = match['team'].iloc[0] if len(match) else None
        r = projector.project_player(
            p, f['opponent'], f['venue'], f['is_home'],
            weather=weather_map.get(f['venue'],'fine'),
            injury_override=injury_map.get(p),
            tog_override=tog_map.get(p),
            factor_weights=factor_weights,
            position=pos, team=team,
            manual_base_scores=manual_base_scores,
        )
        if r: rows.append(r)

    df_proj = pd.DataFrame(rows).sort_values('projection',ascending=False).reset_index(drop=True)
    df_proj.index += 1

    df_proj['role_factor'] = df_proj['player'].map(role_factors).fillna(1.0)
    df_proj['projection']  = (df_proj['projection']*df_proj['role_factor']).round(1)
    df_proj['floor']       = (df_proj['floor']     *df_proj['role_factor']).round(1)
    df_proj['ceiling']     = (df_proj['ceiling']   *df_proj['role_factor']).round(1)

    if 'salary' in ds_players.columns:
        sal = ds_players.set_index('ds_name')['salary'].to_dict()
        df_proj['salary'] = df_proj['player'].map(sal)
        df_proj['value']  = (df_proj['projection']/(df_proj['salary']/1000)).round(2)
        df_proj = df_proj[df_proj['salary'].notna()].reset_index(drop=True)
        df_proj.index += 1

    opp_stat_ratings = build_opp_stat_ratings(df_stats)
    named     = set(df_proj['player'].tolist())
    stat_rows = []
    for player in df_stats['name'].unique():
        if player not in named: continue
        pd_p = df_stats[df_stats['name']==player]
        pt   = pd_p['team'].iloc[-1] if len(pd_p) else None
        if pt not in team_fix: continue
        f = team_fix[pt]
        r = projector.project_stat(
            player, f['opponent'], f['is_home'],
            weather=weather_map.get(f['venue'],'fine'),
            injury_override=injury_map.get(player),
            tog_override=tog_map.get(player),
            factor_weights=factor_weights,
            opp_stat_ratings=opp_stat_ratings,
        )
        if r:
            row = {'player':r['player'],'position':r['position']}
            for stat,prefix in [('disposals','disp'),('kicks','kick'),('handballs','hb'),
                                 ('marks','mark'),('tackles','tackle'),('hit_outs','ho')]:
                d = r[stat]
                row[f'{prefix}_proj']    = d['proj'];   row[f'{prefix}_mean']    = d['mean']
                row[f'{prefix}_floor']   = d['floor'];  row[f'{prefix}_ceiling'] = d['ceiling']
                row[f'{prefix}_avg_20']  = d['avg_20']; row[f'{prefix}_avg_5']   = d['avg_5']
                row.update(d['ou'])
            stat_rows.append(row)

    df_stat = pd.DataFrame(stat_rows).sort_values('disp_proj',ascending=False).reset_index(drop=True)
    df_stat.index += 1
    return df_proj, df_stat


# ── WITH/WITHOUT ANALYSIS ─────────────────────────────────────

def calc_with_without(df_stats, missing_player, missing_team, named_players_df):
    """
    For each named teammate, calculate their average fantasy score
    in rounds where missing_player was out vs rounds they played together.
    Uses last 2 seasons only. Returns a sorted DataFrame.
    """
    FINALS_ORDER = {'EF':100,'QF':101,'SF':102,'PF':103,'GF':104}
    def rsort(r):
        r = str(r).strip()
        if r in FINALS_ORDER: return FINALS_ORDER[r]
        try: return int(r)
        except: return 999

    recent_seasons = sorted(df_stats['season'].unique())[-2:]
    recent = df_stats[df_stats['season'].isin(recent_seasons)].copy()
    recent = recent[recent['tog_pct'] >= 0.45]

    # Get all rounds the missing player's team played
    team_rounds = set(
        zip(recent[recent['team'] == missing_team]['season'],
            recent[recent['team'] == missing_team]['round'])
    )

    # Get rounds the missing player played (including low TOG)
    mp_all = df_stats[
        (df_stats['name'] == missing_player) &
        (df_stats['season'].isin(recent_seasons))
    ]
    mp_played = set(zip(mp_all['season'], mp_all['round']))
    mp_low_tog = set(zip(
        mp_all[mp_all['tog_pct'] < 0.35]['season'],
        mp_all[mp_all['tog_pct'] < 0.35]['round']
    ))

    # Out rounds = team played but missing player didn't (or had very low TOG)
    out_rounds = (team_rounds - mp_played) | mp_low_tog

    # Named teammates only
    named_set = set(named_players_df['ds_name'].tolist())

    rows = []
    for _, tm_row in named_players_df.iterrows():
        teammate = tm_row['ds_name']
        if teammate == missing_player: continue

        tm_data = recent[recent['name'] == teammate]
        if len(tm_data) == 0: continue

        # 2026 season avg
        avg_2026_data = df_stats[
            (df_stats['name'] == teammate) &
            (df_stats['season'] == 2026) &
            (df_stats['tog_pct'] >= 0.45)
        ]
        avg_2026 = round(float(avg_2026_data['fantasy_score'].mean()), 1) if len(avg_2026_data) >= 3 else None

        with_scores    = [r['fantasy_score'] for _, r in tm_data.iterrows()
                          if (r['season'], r['round']) not in out_rounds]
        without_scores = [r['fantasy_score'] for _, r in tm_data.iterrows()
                          if (r['season'], r['round']) in out_rounds]

        n_out = len(without_scores)
        sufficient = n_out >= 3

        if sufficient:
            avg_with    = round(float(np.mean(with_scores)), 1) if with_scores else None
            avg_without = round(float(np.mean(without_scores)), 1)
            if avg_with and avg_with > 0:
                diff    = round(avg_without - avg_with, 1)
                diff_pct = round((diff / avg_with) * 100, 1)
            else:
                diff = diff_pct = None
        else:
            avg_with = avg_without = diff = diff_pct = None

        if diff_pct is not None:
            flag = '✅' if diff_pct > 3 else ('🔴' if diff_pct < -3 else '⚪')
        else:
            flag = '—'

        rows.append({
            'ds_name':    teammate,
            'position':   tm_row['position'].split('/')[0],
            'avg_2026':   avg_2026,
            'avg_with':   avg_with,
            'avg_without':avg_without,
            'diff':       diff,
            'diff_pct':   diff_pct,
            'n_out':      n_out,
            'sufficient': sufficient,
            'flag':       flag,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort: sufficient data first, then by avg_2026 descending
    df['_sort_key'] = df['avg_2026'].fillna(0)
    df = df.sort_values(['sufficient', '_sort_key'], ascending=[False, False]).drop(columns='_sort_key')
    return df.reset_index(drop=True)


# ── STREAMLIT UI ──────────────────────────────────────────────

def main():
    st.title("🏉 AFL Fantasy DFS")

    if 'app_initialised' not in st.session_state:
        saved_fw    = load_factor_weights()
        saved_prefs = load_app_prefs()
        st.session_state.factor_weights  = {k: float(saved_fw.get(k, 1.0)) for k in FACTOR_KEYS}
        st.session_state.saved_season    = saved_prefs.get('season', None)
        st.session_state.saved_round     = saved_prefs.get('round',  None)
        st.session_state.app_initialised = True

    for key, default in [
        ('df_stats',           None),
        ('ds_players',         None),
        ('out_players',        []),
        ('fixtures',           []),
        ('weather_map',        {}),
        ('df_proj',            None),
        ('df_stat_proj',       None),
        ('injury_map',         {}),
        ('tog_map',            {}),
        ('manual_scores',      {}),
        ('inflate_set',        set()),
        ('manual_role_boosts', {}),
        ('slate_name',         ''),
        ('saved_slates',       {}),
        ('round_label',        ''),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏉 AFL Fantasy DFS")
        page = st.radio(
            "",
            ["📊 Projections","📋 Results","📈 Stat Lines","🔍 With/Without","🎯 Most X Stat","⚙️ Add Round Data","🏟️ Opponent Ratings"],
            label_visibility="collapsed"
        )

        if 'saved_slates_loaded' not in st.session_state:
            st.session_state.saved_slates        = load_saved_slates()
            st.session_state.saved_slates_loaded = True

        if st.session_state.saved_slates:
            st.markdown("---")
            st.markdown("**Saved slates**")
            for name in st.session_state.saved_slates:
                if st.button(f"📂 {name}", key=f"load_{name}", use_container_width=True):
                    s = st.session_state.saved_slates[name]
                    for k, v in s.items():
                        if isinstance(v, list) and k in ('df_proj','df_stat_proj','ds_players'):
                            st.session_state[k] = pd.DataFrame(v)
                        else:
                            st.session_state[k] = v
                    st.rerun()

    # ══════════════════════════════════════════════════════════
    # PROJECTIONS PAGE
    # ══════════════════════════════════════════════════════════
    if page == "📊 Projections":
        st.header("Generate Projections")

        if st.session_state.df_stats is None:
            df_stats = load_stats()
            if not df_stats.empty:
                st.session_state.df_stats = df_stats

        if st.session_state.df_stats is None or st.session_state.df_stats.empty:
            st.warning("No player stats loaded. Go to **Add Round Data** to scrape stats first.")

        col1, col2 = st.columns([2,1])
        with col1:
            slate_name = st.text_input("Slate name (e.g. Friday R6, Saturday R6)", value=st.session_state.slate_name)
            st.session_state.slate_name = slate_name
        with col2:
            st.session_state.round_label = st.text_input("Round label", value=st.session_state.round_label)

        # ── 1. SELECT ROUND ───────────────────────────────────
        st.subheader("1. Select Round")

        @st.cache_data(show_spinner=False, ttl=3600)
        def load_fixtures():
            sb   = get_supabase()
            resp = sb.table('fixtures').select('*').execute()
            if not resp.data: return pd.DataFrame()
            df = pd.DataFrame(resp.data)
            df['home_team'] = df['Home Team'].map(lambda x: DS_TEAM_MAP.get(x, x))
            df['away_team'] = df['Away Team'].map(lambda x: DS_TEAM_MAP.get(x, x))
            df['venue']     = df['Location'].map(lambda x: FIXTURE_VENUE_MAP.get(x, x))
            return df

        df_fixtures = load_fixtures()
        if not df_fixtures.empty:
            seasons = sorted(df_fixtures['file_year'].unique(), reverse=True)
            saved_season = st.session_state.get('saved_season')
            season_index = seasons.index(saved_season) if saved_season in seasons else 0
            sel_season   = st.selectbox("Season", seasons, index=season_index, key="sel_season")

            df_season = df_fixtures[df_fixtures['file_year'] == sel_season]
            rounds    = sorted(df_season['Round Number'].unique(),
                               key=lambda r: int(r) if str(r).isdigit() else 999)
            saved_round = st.session_state.get('saved_round')
            round_index = rounds.index(saved_round) if saved_round in rounds else 0
            sel_round   = st.selectbox("Round", rounds, index=round_index, key="sel_round")

            if sel_season != st.session_state.get('saved_season') or sel_round != st.session_state.get('saved_round'):
                st.session_state.saved_season = sel_season
                st.session_state.saved_round  = sel_round
                save_app_prefs({'season': sel_season, 'round': sel_round})

            df_round = df_season[df_season['Round Number'] == sel_round]
            fixtures = []
            for _, row in df_round.iterrows():
                fixtures.append({
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'venue':     row['venue'],
                })
            st.session_state.fixtures = fixtures
            games_str = ' · '.join([f"{f['home_team']} vs {f['away_team']}" for f in fixtures])
            st.info(f"{len(fixtures)} games: {games_str}")
        else:
            st.warning("No fixtures loaded. Check Supabase fixtures table.")

        # ── 2. DRAFTSTARS CSV ─────────────────────────────────
        st.subheader("2. Upload Draftstars CSV")
        ds_file = st.file_uploader("Draftstars CSV", type="csv", label_visibility="collapsed")
        if ds_file:
            try:
                players, out_players = parse_draftstars_csv(ds_file.read())
                st.session_state.ds_players         = players
                st.session_state.out_players        = out_players
                st.session_state.inflate_set        = set()
                st.session_state.manual_role_boosts = {}
                st.success(f"✅ {len(players)} named players · {len(out_players)} OUT")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

        # ── 3. WEATHER ────────────────────────────────────────
        if st.session_state.fixtures:
            st.subheader("3. Weather")
            col1, col2 = st.columns([3,1])
            with col1:
                if st.button("🌤️ Fetch weather automatically"):
                    with st.spinner("Fetching weather..."):
                        st.session_state.weather_map = fetch_all_venue_weather(st.session_state.fixtures)
            wmap   = {}
            venues = list({f['venue'] for f in st.session_state.fixtures})
            cols   = st.columns(min(len(venues), 3))
            for i, v in enumerate(venues):
                with cols[i % 3]:
                    current = st.session_state.weather_map.get(v, 'fine')
                    icons   = {'fine':'☀️','light_rain':'🌦️','heavy_rain':'🌧️','wind':'💨'}
                    wmap[v] = st.selectbox(
                        f"{icons.get(current,'?')} {v}",
                        ['fine','light_rain','heavy_rain','wind'],
                        index=['fine','light_rain','heavy_rain','wind'].index(current),
                        key=f"weather_{v}"
                    )
            st.session_state.weather_map = wmap

        # ── 4. FACTOR WEIGHTS ─────────────────────────────────
        st.subheader("4. Factor Weights")
        fw   = {}
        cols = st.columns(4)
        for i, (k, label) in enumerate(zip(FACTOR_KEYS, FACTOR_LABELS)):
            with cols[i % 4]:
                fw[k] = st.slider(
                    label,
                    min_value=0.2,
                    max_value=1.5,
                    value=float(st.session_state.factor_weights.get(k, 1.0)),
                    step=0.05,
                    key=f"fw_{k}"
                )
        if fw != st.session_state.factor_weights:
            st.session_state.factor_weights = fw
            save_factor_weights(fw)

        # ── 5. PLAYER OVERRIDES ───────────────────────────────
        st.subheader("5. Player Overrides")
        oc1, oc2 = st.columns(2)

        with oc1:
            st.markdown("**Injury / Output reduction**")
            inj_player = st.selectbox("Add player", [""] + (
                sorted(st.session_state.ds_players['ds_name'].tolist())
                if st.session_state.ds_players is not None else []
            ), key="inj_select")
            if inj_player and inj_player not in st.session_state.injury_map:
                if st.button("Add to injury list"):
                    st.session_state.injury_map[inj_player] = 0.75
                    st.rerun()
            for player in list(st.session_state.injury_map.keys()):
                c1, c2 = st.columns([3,1])
                with c1:
                    st.session_state.injury_map[player] = st.slider(
                        f"{player}", 0.3, 1.0,
                        st.session_state.injury_map[player], 0.05,
                        key=f"inj_{player}"
                    )
                with c2:
                    if st.button("✕", key=f"rem_inj_{player}"):
                        del st.session_state.injury_map[player]; st.rerun()

        with oc2:
            st.markdown("**TOG overrides**")
            tog_player = st.selectbox("Add player", [""] + (
                sorted(st.session_state.ds_players['ds_name'].tolist())
                if st.session_state.ds_players is not None else []
            ), key="tog_select")
            if tog_player and tog_player not in st.session_state.tog_map:
                if st.button("Add to TOG list"):
                    st.session_state.tog_map[tog_player] = 0.75
                    st.rerun()
            for player in list(st.session_state.tog_map.keys()):
                c1, c2 = st.columns([3,1])
                with c1:
                    st.session_state.tog_map[player] = st.slider(
                        f"{player} TOG", 0.3, 1.0,
                        st.session_state.tog_map[player], 0.05,
                        key=f"tog_{player}"
                    )
                with c2:
                    if st.button("✕", key=f"rem_tog_{player}"):
                        del st.session_state.tog_map[player]; st.rerun()

        # ── DEBUTANTS ─────────────────────────────────────────
        # CHANGE 7: Position-based default (66% of 2026 position average)
        st.markdown("**Debutant / no history players**")
        if st.session_state.ds_players is not None and st.session_state.df_stats is not None:
            known   = set(st.session_state.df_stats['name'].unique())
            unnamed = [
                row for _, row in st.session_state.ds_players.iterrows()
                if row['ds_name'] not in known
            ]

            # Build 2026 position averages for defaults
            df_2026 = st.session_state.df_stats[
                (st.session_state.df_stats['season'] == 2026) &
                (st.session_state.df_stats['tog_pct'] >= 0.45)
            ]
            pos_avgs_2026 = df_2026.groupby('position')['fantasy_score'].mean().to_dict()

            if unnamed:
                st.warning(f"{len(unnamed)} players have no stats history — enter a base score for each:")
                for row in unnamed:
                    p   = row['ds_name']
                    pos = row['position'].split('/')[0]
                    pos_avg = pos_avgs_2026.get(pos, pos_avgs_2026.get('MID', 60))
                    default_score = round(pos_avg * 0.66, 1)
                    c1, c2 = st.columns([3,1])
                    with c1:
                        st.write(f"**{p}** ({pos} — pos avg {round(pos_avg,1)})")
                    with c2:
                        score = st.number_input(
                            "Base", 0.0, 150.0,
                            float(st.session_state.manual_scores.get(p, default_score)),
                            5.0, key=f"deb_{p}"
                        )
                        st.session_state.manual_scores[p] = score
            else:
                st.success("All named players have stats history.")

        with st.expander("Add player manually"):
            deb_col1, deb_col2, deb_col3 = st.columns([2,1,1])
            with deb_col1:
                deb_name = st.text_input("Player name", key="deb_name")
            with deb_col2:
                deb_score = st.number_input("Base score", 0.0, 150.0, 40.0, key="deb_score")
            with deb_col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add") and deb_name:
                    st.session_state.manual_scores[deb_name] = deb_score
                    st.rerun()
        for p, s in list(st.session_state.manual_scores.items()):
            if st.session_state.ds_players is not None:
                known = set(st.session_state.df_stats['name'].unique()) if st.session_state.df_stats is not None else set()
                if p in known: continue
            c1, c2 = st.columns([3,1])
            with c1: st.write(f"{p}: {s}")
            with c2:
                if st.button("✕", key=f"rem_deb_{p}"):
                    del st.session_state.manual_scores[p]; st.rerun()

        # ── 6. ROLE INFLATION ─────────────────────────────────
        # CHANGE 4 & 6: 2026-only display avg, 2-season threshold, with/without table
        if st.session_state.out_players and st.session_state.df_stats is not None:
            st.subheader("6. Role Inflation")

            df_s = st.session_state.df_stats

            # Threshold check: 2 seasons, TOG >= 0.45, avg >= 80
            recent_2s       = sorted(df_s['season'].unique())[-2:]
            recent_2s_data  = df_s[df_s['season'].isin(recent_2s) & (df_s['tog_pct'] >= 0.45) if 'tog_pct' in df_s.columns else df_s['season'].isin(recent_2s)]

            # Display avg: 2026 only, TOG >= 0.45
            df_2026_tog = df_s[(df_s['season'] == 2026) & (df_s['tog_pct'] >= 0.45)]

            significant = []
            for mp in st.session_state.out_players:
                mp_recent = df_s[
                    (df_s['name'] == mp['name']) &
                    (df_s['season'].isin(recent_2s)) &
                    (df_s['tog_pct'] >= 0.45)
                ]
                if len(mp_recent) >= 3 and mp_recent['fantasy_score'].mean() >= 80:
                    significant.append(mp)

            if not significant:
                st.info("No OUT players averaging 80+ (last 2 seasons) this week.")
            else:
                st.markdown("**Step 1 — Select confirmed outs that will inflate teammates**")
                new_inflate_set = set()
                for mp in significant:
                    # Display 2026-only average
                    mp_2026 = df_2026_tog[df_2026_tog['name'] == mp['name']]
                    if len(mp_2026) >= 1:
                        disp_avg = round(float(mp_2026['fantasy_score'].mean()), 1)
                        avg_label = f"2026 avg {disp_avg}"
                    else:
                        mp_rec = df_s[
                            (df_s['name'] == mp['name']) &
                            (df_s['season'].isin(recent_2s)) &
                            (df_s['tog_pct'] >= 0.45)
                        ]
                        disp_avg = round(float(mp_rec['fantasy_score'].mean()), 1) if len(mp_rec) else '–'
                        avg_label = f"recent avg {disp_avg}"

                    checked = mp['name'] in st.session_state.inflate_set
                    if st.checkbox(
                        f"{mp['name']} ({mp['team']} · {mp['position']} · {avg_label})",
                        value=checked,
                        key=f"inf_{mp['name']}"
                    ):
                        new_inflate_set.add(mp['name'])
                st.session_state.inflate_set = new_inflate_set

# Auto-populate same-position teammates for ticked OUT players
            if st.session_state.inflate_set and st.session_state.ds_players is not None:
                for mp_name in st.session_state.inflate_set:
                    mp = next(
                        (p for p in st.session_state.out_players if p['name'] == mp_name), None
                    )
                    if not mp: continue
                    mp_pos = mp['position'].split('/')[0]
                    mp_team = mp['team']
                    # Find same-position teammates in slate
                    same_pos = st.session_state.ds_players[
                        (st.session_state.ds_players['team'] == mp_team) &
                        (st.session_state.ds_players['position'].str.contains(mp_pos)) &
                        (st.session_state.ds_players['ds_name'] != mp_name)
                    ]['ds_name'].tolist()
                    for p in same_pos:
                        if p not in st.session_state.manual_role_boosts:
                            st.session_state.manual_role_boosts[p] = 1.0

            # Boost sliders
            if st.session_state.ds_players is not None:
                st.markdown("**Manual boosts**")
                st.caption("Use the With/Without page to research teammate impacts.")

                # Manual search for any other player
                st.markdown("**Add player manually**")
                boost_player = st.selectbox(
                    "Search player",
                    [""] + sorted(st.session_state.ds_players['ds_name'].tolist()),
                    key="boost_select"
                )
                if boost_player and boost_player not in st.session_state.manual_role_boosts:
                    st.session_state.manual_role_boosts[boost_player] = 1.0

                # Render all sliders after add logic
                if st.session_state.manual_role_boosts:
                    st.markdown("**Boosts**")
                    for player in list(st.session_state.manual_role_boosts.keys()):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            current_pct = round((st.session_state.manual_role_boosts[player] - 1.0) * 100)
                            boost_pct   = st.slider(
                                f"{player}",
                                min_value=-10,
                                max_value=40,
                                value=int(current_pct),
                                step=2,
                                format="%d%%",
                                key=f"boost_{player}"
                            )
                            st.session_state.manual_role_boosts[player] = round(1.0 + boost_pct / 100, 4)
                        with c2:
                            if st.button("✕", key=f"rem_boost_{player}"):
                                del st.session_state.manual_role_boosts[player]
                                st.rerun()

                    if st.button("🔄 Reset all boosts"):
                        st.session_state.manual_role_boosts = {}
                        st.rerun()

        # ── RUN / SAVE ────────────────────────────────────────
        st.markdown("---")
        col1, col2 = st.columns([2,1])
        with col1:
            run_btn  = st.button("🚀 Run Projections", type="primary", use_container_width=True)
        with col2:
            save_btn = st.button("💾 Save Slate", use_container_width=True)

        if run_btn:
            if st.session_state.df_stats is None or st.session_state.df_stats.empty:
                st.error("No stats loaded. Scrape round data first.")
            elif st.session_state.ds_players is None:
                st.error("Upload a Draftstars CSV first.")
            elif not st.session_state.fixtures:
                st.error("No fixtures found.")
            else:
                with st.spinner("Running projections..."):
                    df_proj, df_stat = run_projections(
                        st.session_state.df_stats,
                        st.session_state.ds_players,
                        st.session_state.fixtures,
                        st.session_state.weather_map,
                        st.session_state.injury_map,
                        st.session_state.tog_map,
                        fw,
                        st.session_state.manual_scores,
                        st.session_state.manual_role_boosts,
                    )
                st.session_state.df_proj      = df_proj
                st.session_state.df_stat_proj = df_stat
                st.success(f"✅ {len(df_proj)} players projected")

        if save_btn and st.session_state.slate_name:
            save_slate_to_supabase(st.session_state.slate_name, {
                'df_proj':            st.session_state.df_proj,
                'df_stat_proj':       st.session_state.df_stat_proj,
                'ds_players':         st.session_state.ds_players,
                'fixtures':           st.session_state.fixtures,
                'weather_map':        st.session_state.weather_map,
                'out_players':        st.session_state.out_players,
                'round_label':        st.session_state.round_label,
                'manual_role_boosts': st.session_state.manual_role_boosts,
            })
            st.success(f"Slate '{st.session_state.slate_name}' saved!")
            st.session_state.saved_slates = load_saved_slates()
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # RESULTS PAGE
    # ══════════════════════════════════════════════════════════
    elif page == "📋 Results":
        st.header("Projection Results")

        if st.session_state.df_proj is None or st.session_state.df_proj.empty:
            st.info("Run projections first.")
            return

        df = st.session_state.df_proj.copy()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            search = st.text_input("🔍 Search player", "")
        with col2:
            pos_filter = st.selectbox("Position", ["All","MID","DEF","FWD","RUC"])
        with col3:
            teams = ["All"] + sorted(st.session_state.df_proj['team'].dropna().unique().tolist())
            team_filter = st.selectbox("Team", teams)
        with col4:
            sort_by = st.selectbox("Sort by", ["projection","value","ceiling","floor","confidence"])

        if search:
            df = df[df['player'].str.contains(search, case=False)]
        if pos_filter != "All":
            df = df[df['position']==pos_filter]
        if team_filter != "All":
            df = df[df['team']==team_filter]
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
        df.index += 1

        display_cols = ['player','team','position','opponent','projection','floor','ceiling','confidence','variance']
        if 'salary' in df.columns:
            display_cols = ['player','team','position','opponent','salary','projection','floor','ceiling','confidence','value']
        if 'role_factor' in df.columns and (df['role_factor'] != 1.0).any():
            display_cols.append('role_factor')

        st.dataframe(df[display_cols], use_container_width=True, height=500)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        rl = st.session_state.round_label or 'Current'

        with col1:
            csv1 = df.to_csv(index=False).encode()
            st.download_button("📥 Export projections CSV", csv1, f"AFL_Projections_{rl}.csv", "text/csv")

        with col2:
            if st.session_state.df_stat_proj is not None:
                csv2 = st.session_state.df_stat_proj.to_csv(index=False).encode()
                st.download_button("📥 Export stat projections CSV", csv2, f"AFL_Stat_Proj_{rl}.csv", "text/csv")

        with col3:
            if st.session_state.df_stat_proj is not None:
                buf  = io.BytesIO()
                df_s = st.session_state.df_stat_proj
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    for sheet, proj_col, mean_col, lines, prefix in [
                        ('Disposals','disp_proj','disp_mean',DISPOSAL_LINES,'disposals'),
                        ('Kicks',    'kick_proj','kick_mean', KICK_LINES,   'kicks'),
                        ('Handballs','hb_proj',  'hb_mean',   HANDBALL_LINES,'handballs'),
                        ('Marks',    'mark_proj','mark_mean', MARK_LINES,   'marks'),
                        ('Tackles',  'tackle_proj','tackle_mean',TACKLE_LINES,'tackles'),
                        ('Hit Outs', 'ho_proj',  'ho_mean',   HITOUT_LINES, 'hit_outs'),
                    ]:
                        ou_cols = [f'{prefix}_over_{l}' for l in lines if f'{prefix}_over_{l}' in df_s.columns]
                        base    = ['player','position']
                        cols    = [c for c in base+[proj_col,mean_col]+ou_cols if c in df_s.columns]
                        tab     = df_s[cols].copy()
                        rename  = {proj_col:'Median',mean_col:'Mean'}
                        rename.update({f'{prefix}_over_{l}':f'{l}+' for l in lines})
                        tab = tab.rename(columns=rename)
                        for col in [f'{l}+' for l in lines]:
                            if col in tab.columns:
                                tab[col] = (tab[col]*100).round(1)
                        tab.sort_values('Median',ascending=False).reset_index(drop=True).to_excel(
                            writer, sheet_name=sheet, index=True, index_label='Rank'
                        )
                buf.seek(0)
                st.download_button(
                    "📥 Export O/U Excel", buf.read(), f"AFL_OU_{rl}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # ══════════════════════════════════════════════════════════
    # STAT LINES PAGE
    # ══════════════════════════════════════════════════════════
    elif page == "📈 Stat Lines":
        st.header("Stat Lines & O/U")

        if st.session_state.df_proj is None or st.session_state.df_stat_proj is None:
            st.info("Run projections first.")
            return

        players  = st.session_state.df_proj['player'].tolist()
        selected = st.selectbox("Select player", players)

        proj_row = st.session_state.df_proj[st.session_state.df_proj['player']==selected]
        stat_row = st.session_state.df_stat_proj[st.session_state.df_stat_proj['player']==selected]

        if len(proj_row):
            r = proj_row.iloc[0]
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Projection", r['projection'])
            col2.metric("Floor / Ceiling", f"{r['floor']} – {r['ceiling']}")
            col3.metric("Confidence", f"{r['confidence']}%")
            col4.metric("Variance", f"{r['variance']}%")

            if r.get('role_factor', 1.0) != 1.0:
                boost_pct = round((r['role_factor'] - 1.0) * 100)
                st.info(
                    f"Role inflation applied: **{'+' if boost_pct > 0 else ''}{boost_pct}%** "
                    f"(factor {r['role_factor']})"
                )

            st.markdown("**Factors applied**")
            fc1,fc2,fc3 = st.columns(3)
            fc1.metric("Form factor",     r.get('form_factor','–'))
            fc2.metric("Opponent factor", r.get('opp_factor','–'))
            fc3.metric("Venue factor",    r.get('venue_factor','–'))
            fc1.metric("Home/Away",       r.get('home_away_factor','–'))
            fc2.metric("Weather",         r.get('weather_factor','–'))
            fc3.metric("TOG factor",      r.get('tog_factor','–'))

        if len(stat_row):
            r = stat_row.iloc[0]
            st.markdown("---")
            st.markdown("**Last 10 games**")
            if st.session_state.df_stats is not None:
                FINALS_ORDER = {'EF':100,'QF':101,'SF':102,'PF':103,'GF':104}
                def rsort(x):
                    x = str(x).strip()
                    if x in FINALS_ORDER: return FINALS_ORDER[x]
                    try: return int(x)
                    except: return 999
                hist = st.session_state.df_stats[st.session_state.df_stats['name']==selected].copy()
                hist['_rs'] = hist['round'].map(rsort)
                hist = hist.sort_values(['season','_rs']).drop(columns='_rs')
                hist = hist[hist['tog_pct'] >= 0.45].tail(10)
                hist = hist[['season','round','opponent','venue','fantasy_score',
                             'kicks','handballs','marks','tackles','hit_outs','tog_pct']].copy()
                hist['tog_pct'] = (hist['tog_pct']*100).round(0).astype(int).astype(str)+'%'
                hist = hist.rename(columns={
                    'fantasy_score':'Score','kicks':'K','handballs':'HB',
                    'marks':'M','tackles':'T','hit_outs':'HO','tog_pct':'TOG'
                })
                st.dataframe(hist.reset_index(drop=True), use_container_width=True, hide_index=True)

            stat_display = []
            for stat, prefix, label in [
                ('disposals','disp','Disposals'),('kicks','kick','Kicks'),
                ('handballs','hb','Handballs'),('marks','mark','Marks'),
                ('tackles','tackle','Tackles'),('hit_outs','ho','Hit Outs'),
            ]:
                if f'{prefix}_proj' in r:
                    stat_display.append({
                        'Stat':          label,
                        'Proj (median)': r[f'{prefix}_proj'],
                        'Mean':          r.get(f'{prefix}_mean','–'),
                        'Floor':         r.get(f'{prefix}_floor','–'),
                        'Ceiling':       r.get(f'{prefix}_ceiling','–'),
                        '20-game avg':   r.get(f'{prefix}_avg_20','–'),
                        '5-game avg':    r.get(f'{prefix}_avg_5','–'),
                    })
            st.dataframe(pd.DataFrame(stat_display), use_container_width=True, hide_index=True)

            st.markdown("**O/U Probabilities**")
            for stat, prefix, label, lines in [
                ('disposals','disp','Disposals',DISPOSAL_LINES),
                ('kicks','kick','Kicks',KICK_LINES),
                ('handballs','hb','Handballs',HANDBALL_LINES),
                ('marks','mark','Marks',MARK_LINES),
                ('tackles','tackle','Tackles',TACKLE_LINES),
                ('hit_outs','ho','Hit Outs',HITOUT_LINES),
            ]:
                ou_cols = [f'{stat}_over_{l}' for l in lines if f'{stat}_over_{l}' in r]
                if not ou_cols: continue
                with st.expander(label):
                    ou_data = {
                        f'{l}+': f"{r[f'{stat}_over_{l}']*100:.1f}%"
                        for l in lines if f'{stat}_over_{l}' in r
                    }
                    st.dataframe(pd.DataFrame([ou_data]), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # WITH/WITHOUT PAGE
    # ══════════════════════════════════════════════════════════
    elif page == "🔍 With/Without":
        st.header("With/Without Analysis")
        st.markdown("Research how teammates perform when a player is out. Use this to inform manual boosts on the Projections page.")

        df_stats = st.session_state.df_stats if st.session_state.df_stats is not None else load_stats()
        if df_stats is None or df_stats.empty:
            st.info("No stats loaded yet.")
        else:
            teams      = sorted(df_stats['team'].dropna().unique().tolist())
            sel_team   = st.selectbox("Select team", teams, key="ww_team")
            team_players = sorted(df_stats[df_stats['team']==sel_team]['name'].unique().tolist())
            sel_player = st.selectbox("Select missing player", team_players, key="ww_player")

            if sel_player:
                recent_2s = sorted(df_stats['season'].unique())[-2:]

                # Out rounds
                team_rounds = set(zip(
                    df_stats[(df_stats['team']==sel_team) & (df_stats['season'].isin(recent_2s))]['season'],
                    df_stats[(df_stats['team']==sel_team) & (df_stats['season'].isin(recent_2s))]['round']
                ))
                mp_all = df_stats[(df_stats['name']==sel_player) & (df_stats['season'].isin(recent_2s))]
                mp_played  = set(zip(mp_all['season'], mp_all['round']))
                mp_low_tog = set(zip(
                    mp_all[mp_all['tog_pct']<0.35]['season'],
                    mp_all[mp_all['tog_pct']<0.35]['round']
                ))
                out_rounds = (team_rounds - mp_played) | mp_low_tog

                # 2026 avg for missing player
                mp_2026 = df_stats[
                    (df_stats['name']==sel_player) & (df_stats['season']==2026) & (df_stats['tog_pct']>=0.45)
                ]
                mp_avg_2026 = round(float(mp_2026['fantasy_score'].mean()), 1) if len(mp_2026)>=1 else '–'
                st.markdown(f"**{sel_player}** · {sel_team} · 2026 avg {mp_avg_2026} · {len(out_rounds)} out rounds (2025–2026)")

                if len(out_rounds) < 3:
                    st.warning(f"Only {len(out_rounds)} out round(s) available in last 2 seasons — insufficient data for reliable analysis.")
                else:
                    recent = df_stats[
                        (df_stats['season'].isin(recent_2s)) & (df_stats['tog_pct']>=0.45)
                    ].copy()

                    teammates = [p for p in df_stats[df_stats['team']==sel_team]['name'].unique() if p != sel_player]
                    rows = []
                    for teammate in teammates:
                        tm_data = recent[recent['name']==teammate]
                        if len(tm_data)==0: continue

                        avg_2026_data = df_stats[
                            (df_stats['name']==teammate) & (df_stats['season']==2026) & (df_stats['tog_pct']>=0.45)
                        ]
                        avg_2026 = round(float(avg_2026_data['fantasy_score'].mean()),1) if len(avg_2026_data)>=3 else None

                        with_scores    = [r['fantasy_score'] for _,r in tm_data.iterrows()
                                          if (r['season'],r['round']) not in out_rounds]
                        without_scores = [r['fantasy_score'] for _,r in tm_data.iterrows()
                                          if (r['season'],r['round']) in out_rounds]
                        n_out      = len(without_scores)
                        sufficient = n_out >= 3

                        if sufficient and with_scores:
                            avg_with    = round(float(np.mean(with_scores)),1)
                            avg_without = round(float(np.mean(without_scores)),1)
                            diff        = round(avg_without - avg_with, 1)
                            diff_pct    = round((diff/avg_with)*100,1) if avg_with>0 else None
                            flag        = '✅' if diff_pct and diff_pct>3 else ('🔴' if diff_pct and diff_pct<-3 else '⚪')
                        else:
                            avg_with=avg_without=diff=diff_pct=None
                            flag='—'

                        pos = df_stats[df_stats['name']==teammate]['position'].iloc[-1] if len(df_stats[df_stats['name']==teammate]) else 'MID'
                        rows.append({
                            'Player':      teammate,
                            'Pos':         pos.split('/')[0],
                            '2026 avg':    avg_2026 if avg_2026 is not None else None,
                            'With avg':    avg_with if avg_with is not None else None,
                            'Without avg': avg_without if avg_without is not None else None,
                            'Diff':        diff if diff is not None else None,
                            'Diff %':      diff_pct if diff_pct is not None else None,
                            'Games out':   n_out,
                            '':            flag,
                            '_sort':       avg_2026 if avg_2026 is not None else 0,
                            '_suf':        sufficient,
                        })

                    df_ww = pd.DataFrame(rows)
                    df_ww = df_ww.sort_values(['_suf','_sort'], ascending=[False,False])
                    df_ww = df_ww.drop(columns=['_sort','_suf']).reset_index(drop=True)
                    st.dataframe(
                        df_ww,
                        use_container_width=True,
                        hide_index=True,
                        height=600,
                        column_config={
                            '2026 avg':    st.column_config.NumberColumn('2026 avg', format="%.1f"),
                            'With avg':    st.column_config.NumberColumn('With avg', format="%.1f"),
                            'Without avg': st.column_config.NumberColumn('Without avg', format="%.1f"),
                            'Diff':        st.column_config.NumberColumn('Diff', format="%.1f"),
                            'Diff %':      st.column_config.NumberColumn('Diff %', format="%.1f%%"),
                        }
                    )
                    st.caption("✅ boosted >3% · ⚪ neutral · 🔴 down >3% · blank = fewer than 3 out rounds")

    # ══════════════════════════════════════════════════════════
    # MOST X STAT PAGE
    # ══════════════════════════════════════════════════════════
    elif page == "🎯 Most X Stat":
        st.header("🎯 Most X Stat")
        st.markdown(
            "Model bookmaker **Most X Stat** markets — select a group of players and a stat, "
            "then see each player's win probability and the implied fair odds vs the bookie price."
        )

        # ── Load data ─────────────────────────────────────────
        df_stats = st.session_state.df_stats
        if df_stats is None:
            df_stats = load_stats()
            if not df_stats.empty:
                st.session_state.df_stats = df_stats

        if df_stats is None or df_stats.empty:
            st.warning("No player stats loaded. Run **Add Round Data** first.")
            st.stop()

        # ── Fixtures needed for opponent/venue/weather ─────────
        fixtures = st.session_state.get('fixtures', [])
        weather_map = st.session_state.get('weather_map', {})
        injury_map  = st.session_state.get('injury_map', {})
        tog_map     = st.session_state.get('tog_map', {})
        factor_weights = st.session_state.get('factor_weights', {})

        team_fix = {}
        for f in fixtures:
            team_fix[f['home_team']] = {'opponent': f['away_team'], 'venue': f['venue'], 'is_home': True}
            team_fix[f['away_team']] = {'opponent': f['home_team'], 'venue': f['venue'], 'is_home': False}

        # ── Stat selector ──────────────────────────────────────
        MOST_STATS = {
            'Disposals':  'disposals',
            'Kicks':      'kicks',
            'Handballs':  'handballs',
            'Marks':      'marks',
            'Tackles':    'tackles',
            'Hit Outs':   'hit_outs',
            'Goals':      'goals',
        }

        col_stat, col_sim = st.columns([2, 1])
        with col_stat:
            stat_label = st.selectbox("Stat", list(MOST_STATS.keys()))
        with col_sim:
            n_sims = st.selectbox("Simulations", [10_000, 50_000, 100_000], index=1,
                                  help="More sims = more accurate probabilities, slightly slower")

        stat_key = MOST_STATS[stat_label]

        # ── Player selector ────────────────────────────────────
        all_names = sorted(df_stats['name'].unique().tolist())
        # Prefer players in the current slate if loaded
        ds_players = st.session_state.get('ds_players')
        if ds_players is not None and not ds_players.empty and 'ds_name' in ds_players.columns:
            slate_names = sorted(ds_players['ds_name'].tolist())
            default_pool = slate_names
        else:
            default_pool = all_names

        st.markdown("**Select players in the group** (2–8 players)")
        selected_players = st.multiselect(
            "Players",
            options=default_pool,
            label_visibility="collapsed",
            placeholder="Type to search players…",
        )

        # Allow searching outside slate
        if ds_players is not None and not ds_players.empty:
            extra = st.multiselect(
                "Add players not in slate",
                options=[n for n in all_names if n not in default_pool],
                label_visibility="visible",
            )
            selected_players = selected_players + extra

        if len(selected_players) < 2:
            st.info("Select at least 2 players to model the market.")
            st.stop()

        if len(selected_players) > 8:
            st.warning("Maximum 8 players supported. Please remove some.")
            st.stop()

        st.markdown("---")

        # ── Build per-player projections ───────────────────────
        projector        = AFLFantasyProjector(df_stats)
        opp_stat_ratings = build_opp_stat_ratings(df_stats)

        player_data = []  # list of dicts: name, proj, std, floor, ceiling, games_n

        for pname in selected_players:
            pd_p = df_stats[df_stats['name'] == pname]
            pt   = pd_p['team'].iloc[-1] if len(pd_p) else None
            fix  = team_fix.get(pt, {})

            opponent   = fix.get('opponent', 'Unknown')
            venue      = fix.get('venue', 'Unknown')
            is_home    = fix.get('is_home', False)
            weather    = weather_map.get(venue, 'fine')

            proj_val = std_val = floor_val = ceil_val = avg_5 = avg_20 = None
            games_n  = 0

            # Goals uses simple mean/std (not in project_stat)
            if stat_key == 'goals':
                pd_tog = pd_p[pd_p['tog_pct'] >= 0.45].copy()
                if len(pd_tog) >= 3:
                    r20    = pd_tog['goals'].tail(20).values
                    r5     = pd_tog['goals'].tail(5).values
                    base   = wavg(r20)
                    med_raw = float(np.median(winsorise(r20)))
                    form_f = float(np.clip(r5.mean() / base, 0.80, 1.20)) if base > 0 else 1.0
                    tf     = calc_trend(r20)
                    wmap_g = {'fine': 1.00, 'light_rain': 0.95, 'heavy_rain': 0.88, 'wind': 0.93}
                    wf     = wmap_g.get(weather, 1.00)
                    adj_form = 0.60 + 0.40 * (1.0 + (form_f - 1.0) * factor_weights.get('form', 1.0))
                    adj_tr   = 0.80 + 0.20 * (1.0 + (tf     - 1.0) * factor_weights.get('trend', 1.0))
                    adj_wf   = 1.0 + (wf - 1.0) * factor_weights.get('weather', 1.0)
                    proj_val  = round(max(0, med_raw * adj_form * adj_tr * adj_wf), 2)
                    std_val   = float(pd_tog['goals'].tail(10).std() or proj_val * 0.50)
                    floor_val = round(max(0, proj_val - 1.5 * std_val), 1)
                    ceil_val  = round(proj_val + 1.5 * std_val, 1)
                    avg_5     = round(float(r5.mean()), 1)
                    avg_20    = round(float(base), 1)
                    games_n   = len(r20)
            else:
                if fix and pt in team_fix:
                    stat_proj = projector.project_stat(
                        pname, opponent, is_home, weather,
                        injury_override=injury_map.get(pname),
                        tog_override=tog_map.get(pname),
                        factor_weights=factor_weights,
                        opp_stat_ratings=opp_stat_ratings,
                    )
                    if stat_proj and stat_key in stat_proj:
                        d         = stat_proj[stat_key]
                        proj_val  = d['proj']
                        std_val   = d['std']
                        floor_val = d['floor']
                        ceil_val  = d['ceiling']
                        avg_5     = d['avg_5']
                        avg_20    = d['avg_20']
                        pd_tog    = pd_p[pd_p['tog_pct'] >= 0.45]
                        games_n   = min(20, len(pd_tog))

                # Fallback: use raw stats if no fixture context
                if proj_val is None:
                    pd_tog = pd_p[pd_p['tog_pct'] >= 0.45].copy()
                    if stat_key in pd_tog.columns and len(pd_tog) >= 3:
                        r20      = pd_tog[stat_key].tail(20).values
                        r5_vals  = pd_tog[stat_key].tail(5).values
                        proj_val = round(float(np.median(winsorise(r20))), 1)
                        std_val  = float(pd_tog[stat_key].tail(10).std() or proj_val * 0.30)
                        floor_val = round(max(0, proj_val - 1.5 * std_val), 1)
                        ceil_val  = round(proj_val + 1.5 * std_val, 1)
                        avg_5    = round(float(r5_vals.mean()), 1)
                        avg_20   = round(float(wavg(r20)), 1)
                        games_n  = len(r20)

            player_data.append({
                'name':    pname,
                'team':    pt or '—',
                'proj':    proj_val,
                'std':     std_val,
                'floor':   floor_val,
                'ceiling': ceil_val,
                'avg_5':   avg_5,
                'avg_20':  avg_20,
                'games_n': games_n,
                'opponent': opponent,
                'no_data': proj_val is None,
            })

        # Warn about missing projections
        missing = [p['name'] for p in player_data if p['no_data']]
        if missing:
            st.warning(f"No stat data found for: {', '.join(missing)}. They'll be excluded from simulation.")
        valid = [p for p in player_data if not p['no_data']]

        if len(valid) < 2:
            st.error("Need at least 2 players with data to run simulation.")
            st.stop()

        # ── Monte Carlo simulation ─────────────────────────────
        rng = np.random.default_rng(42)

        # Goals are non-negative integers — use Poisson; others use truncated normal
        if stat_key == 'goals':
            # Poisson for discrete goal counts
            samples = np.column_stack([
                rng.poisson(lam=max(p['proj'], 0.01), size=n_sims)
                for p in valid
            ])
        else:
            samples = np.column_stack([
                np.clip(
                    rng.normal(loc=p['proj'], scale=max(p['std'], 0.5), size=n_sims),
                    0, None
                )
                for p in valid
            ])

        # Winner = player with highest value; ties split equally
        winners     = samples.argmax(axis=1)           # index of winner per sim
        tie_mask    = (samples == samples.max(axis=1, keepdims=True)).sum(axis=1) > 1
        win_counts  = np.zeros(len(valid))

        for i in range(len(valid)):
            solo_wins = np.sum((winners == i) & ~tie_mask)
            tie_wins  = np.sum(tie_mask & (samples[:, i] == samples.max(axis=1))) / \
                        (samples == samples.max(axis=1, keepdims=True)).sum(axis=1)[
                            tie_mask & (samples[:, i] == samples.max(axis=1))
                        ].mean() if np.any(tie_mask & (samples[:, i] == samples.max(axis=1))) else 0
            win_counts[i] = solo_wins + tie_wins

        win_probs = win_counts / n_sims

        # Normalise to 1.0 (floating point safety)
        if win_probs.sum() > 0:
            win_probs = win_probs / win_probs.sum()

        # ── Results table ──────────────────────────────────────
        st.subheader(f"Most {stat_label} — Group Results")

        # Bookie odds inputs
        st.markdown("**Enter bookmaker odds (decimal) — leave blank if unknown**")
        bookie_odds = {}
        bookie_cols = st.columns(len(valid))
        for i, p in enumerate(valid):
            with bookie_cols[i]:
                val = st.number_input(
                    p['name'].split()[-1],  # surname only to save space
                    min_value=1.01, max_value=50.0, value=None,
                    step=0.05, format="%.2f",
                    key=f"odds_{p['name']}_{stat_key}",
                    label_visibility="visible",
                )
                bookie_odds[p['name']] = val

        st.markdown("---")

        # Build results dataframe
        rows_out = []
        for i, p in enumerate(valid):
            wp         = win_probs[i]
            impl_odds  = round(1 / wp, 2) if wp > 0 else None
            bk         = bookie_odds.get(p['name'])
            if bk and impl_odds:
                edge = round((bk / impl_odds - 1) * 100, 1)
                edge_str = f"+{edge}%" if edge > 0 else f"{edge}%"
                value_flag = "✅ VALUE" if edge >= 5 else ("⚠️ marginal" if edge >= 1 else "❌ overpriced")
            else:
                edge_str   = "—"
                value_flag = "—"

            rows_out.append({
                'Player':       p['name'],
                'Team':         p['team'],
                'Opp':          p['opponent'],
                f'Proj {stat_label}': p['proj'],
                'Avg 5':        p['avg_5'],
                'Avg 20':       p['avg_20'],
                'Floor':        p['floor'],
                'Ceiling':      p['ceiling'],
                'Games (n)':    p['games_n'],
                'Win %':        round(wp * 100, 1),
                'Fair Odds':    impl_odds,
                'Bookie Odds':  bk,
                'Edge':         edge_str,
                'Value':        value_flag,
            })

        df_out = pd.DataFrame(rows_out).sort_values('Win %', ascending=False).reset_index(drop=True)

        # Colour-code value column
        def colour_value(val):
            if '✅' in str(val): return 'background-color:#d4edda;color:#155724;font-weight:bold'
            if '⚠️' in str(val): return 'background-color:#fff3cd;color:#856404'
            if '❌' in str(val): return 'background-color:#f8d7da;color:#721c24'
            return ''

        def colour_winpct(val):
            try:
                v = float(val)
                if v >= 40: return 'background-color:#d4edda;color:#155724'
                if v >= 25: return 'background-color:#fff3cd;color:#856404'
                return ''
            except: return ''

        st.dataframe(
            df_out.style
                .applymap(colour_value, subset=['Value'])
                .applymap(colour_winpct, subset=['Win %']),
            use_container_width=True,
            hide_index=True,
            column_config={
                f'Proj {stat_label}': st.column_config.NumberColumn(f'Proj {stat_label}', format="%.1f"),
                'Avg 5':              st.column_config.NumberColumn('Avg 5', format="%.1f"),
                'Avg 20':             st.column_config.NumberColumn('Avg 20', format="%.1f"),
                'Floor':              st.column_config.NumberColumn('Floor', format="%.1f"),
                'Ceiling':            st.column_config.NumberColumn('Ceiling', format="%.1f"),
                'Win %':              st.column_config.NumberColumn('Win %', format="%.1f%%"),
                'Fair Odds':          st.column_config.NumberColumn('Fair Odds', format="%.2f"),
                'Bookie Odds':        st.column_config.NumberColumn('Bookie Odds', format="%.2f"),
            }
        )

        # ── Win probability bar chart ──────────────────────────
        st.markdown("#### Win probability breakdown")
        chart_df = df_out[['Player', 'Win %']].copy()
        chart_df['Label'] = chart_df.apply(
            lambda r: f"{r['Player'].split()[-1]}\n{r['Win %']:.1f}%", axis=1
        )
        st.bar_chart(chart_df.set_index('Player')['Win %'])

        # ── Simulation distribution (optional expander) ────────
        with st.expander("📊 Show score distribution per player"):
            import altair as alt
            dist_rows = []
            for i, p in enumerate(valid):
                samp = samples[:, i]
                # bin into histogram buckets
                counts, edges = np.histogram(samp, bins=30)
                for c, e in zip(counts, edges[:-1]):
                    dist_rows.append({
                        'Player': p['name'].split()[-1],
                        stat_label: round(e, 1),
                        'Count': int(c),
                    })
            df_dist = pd.DataFrame(dist_rows)
            chart = (
                alt.Chart(df_dist)
                .mark_bar(opacity=0.6)
                .encode(
                    x=alt.X(f'{stat_label}:Q', bin=False, title=stat_label),
                    y=alt.Y('Count:Q'),
                    color=alt.Color('Player:N'),
                    tooltip=['Player', stat_label, 'Count'],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        st.caption(
            f"Win % calculated via {n_sims:,}-simulation Monte Carlo. "
            f"{'Poisson distribution (discrete count)' if stat_key == 'goals' else 'Truncated normal distribution'}. "
            "Fair Odds = 1 / Win%. Edge = (Bookie ÷ Fair − 1) × 100. "
            "✅ VALUE = edge ≥ 5% · ⚠️ marginal = 1–5% · ❌ overpriced = negative edge."
        )

    # ══════════════════════════════════════════════════════════
    # ADD ROUND DATA PAGE
    # ══════════════════════════════════════════════════════════
    elif page == "⚙️ Add Round Data":
        st.header("Add Round Data")
        st.info("Scrapes AFL Tables for all players. Takes 5–10 minutes. Run once after the round completes.")

        df_roster = load_roster()
        if df_roster.empty:
            st.warning("No player roster found.")
            uploaded_roster = st.file_uploader("Upload player_roster.csv", type="csv")
            if uploaded_roster:
                df_roster = pd.read_csv(uploaded_roster)
                save_roster_to_supabase(df_roster)
                st.success("Roster saved!")
                st.rerun()
            return

        col1, col2 = st.columns(2)
        with col1:
            season = st.selectbox("Season", [2026,2025,2024,2023])
        with col2:
            round_num = st.number_input("Round number (AFL Tables)", 1, 30, 1)

        if st.button("🔄 Start scrape", type="primary"):
            with st.spinner("Loading existing stats..."):
                df_stats = load_stats()
            with st.spinner("Building venue lookup..."):
                sb = get_supabase()
                fix_resp = sb.table('fixtures').select('*').execute()
                venue_lookup   = {}
                is_home_lookup = {}
                if fix_resp.data:
                    for row in fix_resp.data:
                        home = DS_TEAM_MAP.get(row.get('Home Team',''), row.get('Home Team',''))
                        away = DS_TEAM_MAP.get(row.get('Away Team',''), row.get('Away Team',''))
                        venue = FIXTURE_VENUE_MAP.get(row.get('Location',''), row.get('Location',''))
                        date_str = row.get('Date','')
                        try:
                            game_date = datetime.strptime(date_str.split(' ')[0], '%d/%m/%Y').date()
                        except:
                            continue
                        venue_lookup[(home, away, game_date)] = venue
                        venue_lookup[(away, home, game_date)] = venue
                        is_home_lookup[(home, away, game_date)] = True
                        is_home_lookup[(away, home, game_date)] = False

            log_area  = st.empty()
            log_lines = []
            existing_keys = set()
            if not df_stats.empty:
                for _, row in df_stats.iterrows():
                    existing_keys.add((row['name'], int(row['season']), str(row['round'])))

            new_records = []
            players     = df_roster.to_dict('records')

            for i, player in enumerate(players):
                name = player.get('ds_name') or player.get('name') or player.get('Name','')
                team = player.get('team') or player.get('Team','')
                pos  = player.get('position') or player.get('Position','MID')
                if not name: continue

                key = (name, season, str(round_num))
                if key in existing_keys:
                    log_lines.append(f"[{i+1}/{len(players)}] {name} — already have Rd {round_num}")
                    log_area.code('\n'.join(log_lines[-50:]))
                    continue

                records, status = scrape_with_fallbacks(
                    name, team, pos, [season], venue_lookup, is_home_lookup
                )
                round_recs = [
                    r for r in records
                    if str(r['round'])==str(round_num) and r['season']==season
                ]

                if round_recs:
                    new_records.extend(round_recs)
                    fs = round_recs[0]['fantasy_score']
                    log_lines.append(f"[{i+1}/{len(players)}] {name} ✓  score={fs}")
                else:
                    log_lines.append(f"[{i+1}/{len(players)}] {name} — {status}")
                log_area.code('\n'.join(log_lines[-50:]))

            if new_records:
                save_stats_to_supabase(new_records)
                st.success(f"✅ Added {len(new_records)} records for Round {round_num} {season}")
            else:
                st.warning("No new records found.")

    # ══════════════════════════════════════════════════════════
    # OPPONENT RATINGS PAGE
    # ══════════════════════════════════════════════════════════
    elif page == "🏟️ Opponent Ratings":
        st.header("Opponent Ratings")

        df_stats = st.session_state.df_stats if st.session_state.df_stats is not None else load_stats()
        if df_stats is None or df_stats.empty:
            st.info("No stats loaded yet.")
            return

        projector = AFLFantasyProjector(df_stats)
        rows  = []
        teams = sorted(set(df_stats['opponent'].unique()))
        for team in teams:
            row = {'Team': team}
            for pos in POSITIONS:
                row[f'vs {pos}'] = round(projector.opp_ratings.get(pos,{}).get(team,1.0),3)
            rows.append(row)
        df_opp = pd.DataFrame(rows).sort_values('vs MID',ascending=False).reset_index(drop=True)

        def colour_rating(val):
            if not isinstance(val, float): return ''
            if val > 1.05: return 'background-color:#d4edda;color:#155724'
            if val < 0.95: return 'background-color:#f8d7da;color:#721c24'
            return ''

        st.dataframe(
            df_opp.style.applymap(colour_rating, subset=['vs MID','vs DEF','vs FWD','vs RUC']),
            use_container_width=True, hide_index=True
        )

if __name__ == '__main__':
    main()
