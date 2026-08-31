from __future__ import annotations
import base64
from pathlib import Path

def _pick(folder: Path, candidates: list[str]):
    for name in candidates:
        p=folder/name
        if p.exists() and p.is_file(): return p
    return None

def _uri(path: Path):
    return "data:font/ttf;base64,"+base64.b64encode(path.read_bytes()).decode("ascii")

def iransansx_css(base_dir: Path) -> str:
    folder=Path(base_dir)/"assets"/"fonts"
    regular=_pick(folder,["IRANSansX-Regular.ttf","IRANSansX-Regular(1).ttf","IRANSansXFaNum-Regular.ttf","IRANSansXFaNum-Regular(1).ttf"])
    medium=_pick(folder,["IRANSansX-Medium.ttf","IRANSansX-Medium(1).ttf","IRANSansXFaNum-Medium.ttf","IRANSansXFaNum-Medium(1).ttf"])
    bold=_pick(folder,["IRANSansX-Bold.ttf","IRANSansX-Bold(1).ttf","IRANSansXFaNum-Bold.ttf","IRANSansXFaNum-Bold(1).ttf"])
    if not regular: return ""
    rules=[f"@font-face{{font-family:'IRANSansX';src:url('{_uri(regular)}') format('truetype');font-weight:400;font-display:swap;}}"]
    if medium: rules.append(f"@font-face{{font-family:'IRANSansX';src:url('{_uri(medium)}') format('truetype');font-weight:500 600;font-display:swap;}}")
    if bold: rules.append(f"@font-face{{font-family:'IRANSansX';src:url('{_uri(bold)}') format('truetype');font-weight:700 900;font-display:swap;}}")
    rules.append("html,body,[class*='css'],.stApp,*{font-family:'IRANSansX','Vazirmatn','Segoe UI',Tahoma,sans-serif!important;}")
    return "<style>"+"".join(rules)+"</style>"
