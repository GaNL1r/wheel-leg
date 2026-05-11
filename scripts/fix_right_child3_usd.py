"""Rename Right_front_joint3_joint → Right_front_child3_joint in the USD file.

This must be run with Isaac Sim's Python (which has pxr available):

    python scripts/fix_right_child3_usd.py

It modifies the embedded USD asset in-place and leaves a backup at *.usd.bak.
"""

import shutil
import sys

from pxr import Sdf, Usd


USD_PATH = "source/stackforce_simready_cod_2026robomaster_balance_closed_usd_closed_usd_lab/stackforce_simready_cod_2026robomaster_balance_closed_usd_closed_usd_lab/assets/robots/cod_2026robomaster_balance_closed_usd/usd/COD-2026RoboMaster-Balance.usd"


def main():
    old_name = "Right_front_joint3_joint"
    new_name = "Right_front_child3_joint"

    # Open stage
    stage = Usd.Stage.Open(USD_PATH)
    if not stage:
        print(f"ERROR: Failed to open USD at {USD_PATH}", flush=True)
        sys.exit(1)

    # Find the misnamed prim
    target = None
    for prim in stage.TraverseAll():
        if prim.GetName() == old_name:
            target = prim
            break

    if target is None:
        print(f"Prim '{old_name}' not found — may already be fixed.", flush=True)
        return

    old_path = target.GetPath()
    new_path = Sdf.Path(str(old_path).replace(old_name, new_name))

    print(f"Renaming: {old_path} → {new_path}", flush=True)

    # Backup original
    backup = USD_PATH + ".bak"
    shutil.copy2(USD_PATH, backup)
    print(f"Backup saved to {backup}", flush=True)

    # Reroot the prim (USD rename)
    layer = stage.GetRootLayer()
    layer_edit = Sdf.BatchNamespaceEdit()
    layer_edit.Add(str(old_path), str(new_path))
    if not layer.Apply(layer_edit):
        print("ERROR: BatchNamespaceEdit failed.", flush=True)
        sys.exit(1)

    # Save
    stage.GetRootLayer().Save()
    print(f"Renamed. Saved to {USD_PATH}", flush=True)

    # Verify
    stage2 = Usd.Stage.Open(USD_PATH)
    found_old = False
    found_new = False
    for prim in stage2.TraverseAll():
        if prim.GetName() == old_name:
            found_old = True
        if prim.GetName() == new_name:
            found_new = True
    if found_old:
        print("WARNING: old name still present!", flush=True)
    if found_new:
        print("SUCCESS: new name confirmed.", flush=True)


if __name__ == "__main__":
    main()
