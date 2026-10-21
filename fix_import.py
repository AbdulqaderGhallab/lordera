import pathlib
p = pathlib.Path("controller_service.py")
text = p.read_text()

# Replace the whole block from "# Try Akash SDK" to "SDK_AVAILABLE = False"
import re
pattern = r"# Try Akash SDK.*?SDK_AVAILABLE = False"
replacement = """# Try Akash SDK - official PyPI: pip install akash
try:
    from akash import AkashClient
    try:
        from akash import AkashWallet
    except ImportError:
        AkashWallet = None
    SDL = None  # SDL is handled as YAML string, not a class in this SDK
    SDK_AVAILABLE = True
    print("Akash SDK imported successfully")
except Exception as e:
    try:
        from akash.client import AkashClient
        from akash.wallet import AkashWallet
        SDL = None
        SDK_AVAILABLE = True
        print(f"Akash SDK imported from submodule: {e}")
    except Exception as e2:
        print(f"Akash SDK import failed: {e} | {e2}")
        AkashClient = None
        AkashWallet = None
        SDL = None
        SDK_AVAILABLE = False"""

text_new = re.sub(pattern, replacement, text, flags=re.DOTALL)
if text_new != text:
    p.write_text(text_new)
    print("Fixed!")
else:
    print("Pattern not found, trying manual")
    # fallback manual edit lines 52-61
    lines = text.splitlines()
    new_lines = []
    skip = False
    for i, line in enumerate(lines):
        if "# Try Akash SDK" in line:
            skip = True
            new_lines.append(replacement)
        if skip and "SDK_AVAILABLE = False" in line and i>52:
            # find second occurrence
            if "AkashClient = None" not in "\n".join(new_lines[-5:]):
                continue
            else:
                skip = False
                continue
        if not skip:
            new_lines.append(line)
        if skip and "SDK_AVAILABLE = False" in line:
            # end after second line
            if text.count("SDK_AVAILABLE = False")>1:
                pass
    # just write replacement for simplicity
    print("manual fallback - please edit file manually")

