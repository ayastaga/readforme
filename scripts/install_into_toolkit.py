#!/usr/bin/env python3
"""Install ReadForMe into a cohere-toolkit checkout.

    python scripts/install_into_toolkit.py /path/to/cohere-toolkit

Copies two new files and applies five small, anchored edits. Idempotent: re-running
is a no-op. Tested against cohere-toolkit at commit range Mar-2025 (last upstream commit).
Each edit is verified by an anchor string; if an anchor is missing the script stops
and tells you which file to patch by hand (see docs/toolkit_integration.md).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TK_FILES = HERE / "toolkit" / "src" / "backend"


def edit(path: Path, anchor: str, insert: str, *, before: bool = False, marker: str | None = None) -> bool:
    s = path.read_text()
    marker = marker or insert.strip().splitlines()[0]
    if marker in s:
        print(f"  = {path.relative_to(path.parents[3])}: already patched")
        return False
    if anchor not in s:
        sys.exit(f"!! anchor not found in {path}: {anchor!r}\n   Patch by hand, see docs/toolkit_integration.md")
    s = s.replace(anchor, (insert + anchor) if before else (anchor + insert), 1)
    path.write_text(s)
    print(f"  + {path.relative_to(path.parents[3])}")
    return True


def main(root: Path):
    be = root / "src" / "backend"
    if not (be / "tools" / "base.py").exists():
        sys.exit(f"{root} does not look like a cohere-toolkit checkout")

    print("Copying files")
    for rel in ("tools/read_this_for_me.py", "services/readforme_upload.py"):
        shutil.copy(TK_FILES / rel, be / rel)
        print(f"  + src/backend/{rel}")

    print("Patching")
    # 1. tools/__init__.py – export
    edit(be / "tools" / "__init__.py",
         "from backend.tools.python_interpreter import PythonInterpreter\n",
         "from backend.tools.read_this_for_me import ReadThisForMeTool\n")
    edit(be / "tools" / "__init__.py", '    "PythonInterpreter",\n', '    "ReadThisForMeTool",\n')

    # 2. config/tools.py – register in the Tool enum
    edit(be / "config" / "tools.py", "    PythonInterpreter,\n", "    ReadThisForMeTool,\n",
         marker="    ReadThisForMeTool,\n")
    edit(be / "config" / "tools.py", "    Python_Interpreter = PythonInterpreter\n",
         "    Read_This_For_Me = ReadThisForMeTool\n")

    # 3. config/settings.py – settings model + field on ToolSettings
    edit(be / "config" / "settings.py", "class ToolSettings(BaseSettings, BaseModel):\n",
         '''class ReadThisForMeSettings(BaseSettings, BaseModel):
    model_config = SETTINGS_CONFIG
    url: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("READFORME_URL", "url")
    )
    default_language: Optional[str] = Field(
        default="en", validation_alias=AliasChoices("READFORME_DEFAULT_LANGUAGE", "default_language")
    )
    timeout_s: Optional[float] = Field(
        default=300, validation_alias=AliasChoices("READFORME_TIMEOUT_S", "timeout_s")
    )


''', before=True)
    edit(be / "config" / "settings.py",
         "    python_interpreter: Optional[PythonToolSettings] = Field(\n        default=PythonToolSettings()\n    )\n",
         "    read_this_for_me: Optional[ReadThisForMeSettings] = Field(\n        default=ReadThisForMeSettings()\n    )\n")

    # 4. services/file.py – accept images at upload
    edit(be / "services" / "file.py", "import backend.crud.file as file_crud\n",
         "from backend.services.readforme_upload import is_image_extension, read_image_via_readforme\n")
    edit(be / "services" / "file.py",
         "    if file_extension == PDF_EXTENSION:\n        return utils.read_pdf(file_contents)\n",
         "    if is_image_extension(file_extension):\n        return read_image_via_readforme(file_contents, file.filename)\n", before=True)

    # 5. configuration.template.yaml – document the setting
    edit(be / "config" / "configuration.template.yaml", "  python_interpreter:\n",
         "  read_this_for_me:\n    url: http://readforme:8090\n    default_language: en\n    timeout_s: 300\n", before=True)

    # 6. frontend – accept image mime types
    fe = root / "src" / "interfaces" / "assistants_web" / "src"
    consts = fe / "constants" / "conversation.ts"
    if consts.exists():
        edit(consts, "  'application/pdf',\n", "  'image/jpeg',\n  'image/png',\n  'image/webp',\n")
        edit(fe / "utils" / "file.ts", "      ['png']: 'image/png',\n", "      ['jpg']: 'image/jpeg',\n      ['jpeg']: 'image/jpeg',\n      ['webp']: 'image/webp',\n")

    # 7. compose overlay
    shutil.copy(HERE / "toolkit" / "docker-compose.readforme.yml", root / "docker-compose.readforme.yml")
    print("  + docker-compose.readforme.yml")
    print("\nDone. Next:\n  cp src/backend/config/configuration.template.yaml src/backend/config/configuration.yaml  (if you haven't)\n"
          "  docker compose -f docker-compose.yml -f docker-compose.readforme.yml up --build")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve())
