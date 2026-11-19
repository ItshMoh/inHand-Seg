"""Synthetic Phase-7 arithmetic, provenance, selection, resume and rendering checks.

No real dataset inference, SAM, model downloads or remote calls.
"""

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.clutter_config import DEFAULTS as CLUTTER
from src.clutter_sam import (encode_rle, file_sha256, filter_proposals, fingerprint,
                            image_identity, image_key, read_record)
from src.clutter_sam import write_record
from src.clutter_workflow import proposal_path
from src.phase7_analysis import (CLASSES, code_hashes, describe, image_rows, ious, label_statistics,
                                 sealed, source, source_summaries, strict_json, video_aliases)
from src.phase7_config import DEFAULTS, FAILURE_TAGS, load_config, validate_config
from src.phase7_render import FP, FN, object_errors, panel_arrays, render_case, survey_index
from src.phase7_selection import (build_selection, choose_shortlist, clutter_hints, key,
                                  make_request, review_template, validate_selection)
from src.phase7_workflow import (bare, check_predictions, fetch_files, gt_lookup,
                                 prediction_statistics, preview_token, render_selected,
                                 selected_region, status, survey, validate_approval, verify_request)
from phase7_report import approve_preview

RESULTS = {}


def check(name, ok):
    RESULTS[name] = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def raises(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return True
    return False


def fixture(tmp):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["splits"] = {"test_indomain": 10, "test_outdomain": 2}  # isolated synthetic population
    cfg["execution"]["scan_commit_every"] = 3
    cfg6 = copy.deepcopy(CLUTTER)
    cfg6["data_root"] = str(tmp / "data")
    cfg6["cache_root"] = str(tmp / "sam_cache")
    m6 = {"proposal_fingerprint": "proposal-identity"}
    inventory, evals, originals, predictions, measurements = [], {}, {}, {}, []
    sources = ["ego4d"]*4 + ["epic"]*2 + ["escape"]*2 + ["thu"]*2 + ["youtube"]*2
    for i, src in enumerate(sources):
        split = "test_outdomain" if src == "youtube" else "test_indomain"
        name = f"{src}_video{i//2}_{i:06d}"
        original = np.zeros((40, 60), np.uint8)
        original[14:24,6:13] = 1
        if i != 11:
            original[15:28,20:38] = 3
            original[16,22] = 5
            original[2,2] = 3
        original[:3,-3:] = 7
        truth = np.array([0,1,1,2,2,2,255,255,255], np.uint8)[original]
        rgb = np.full((40,60,3), 30+i*10, np.uint8)
        image_path = Path(cfg6["data_root"])/split/"image"/f"{name}.jpg"
        label_path = image_path.parent.parent/"label"/f"{name}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(original).save(label_path)
        e = {"split": split, "name": name, "width": 60, "height": 40,
             "image_sha256": file_sha256(image_path), "label_sha256": file_sha256(label_path),
             "display_source": source(name), "video": name.rsplit("_",1)[0]}
        inventory.append(e)
        pred = {}
        region = np.zeros(truth.shape, bool)
        region[32:36,45:50] = True
        for system, runs in (("baseline",cfg["baseline_runs"]), ("cascade",cfg["cascade_runs"])):
            for seed, run in enumerate(runs):
                p = np.where(truth==255,0,truth).astype(np.uint8)
                bad = (i % 2 == 0) == (system == "baseline")
                p[15:24 if bad else 17,20:38] = 0
                p[region] = 2 if ((i%4 in (2,3)) == (system=="baseline")) else 0
                if seed and truth[20,30] == 2:
                    p[20,30] = 0
                pred[run] = p
                m = evals.setdefault((run,split,"predicted" if system=="cascade" else None), {"per_image":{}})
                m["per_image"][name] = {"size":[60,40], "iou": prediction_statistics(p,truth)["iou"]}
        proposals = [{"mask_id":0,"segmentation":encode_rle(region),"area":int(region.sum()),"predicted_iou":.99,"stability_score":.99}]
        filtered = filter_proposals(proposals,truth,cfg6["filter"],cfg6["distance_bin_edges"])
        c = filtered["candidates"][0]
        candidate = {k:v for k,v in c.items() if k != "segmentation"}
        candidate["object_fractions"] = {run:float((p[region]==2).mean()) for run,p in pred.items()}
        measured = {**e,"has_hands":True,"candidates":[candidate],"proposal_masks_sha256":fingerprint(proposals),
                    "eligible_masks_sha256":fingerprint(filtered["candidates"])}
        measurements.append(measured)
        write_record(proposal_path(cfg6,m6,e), {"proposal_fingerprint":m6["proposal_fingerprint"],
                     "image":image_identity(e),"proposals":proposals})
        originals[key(e)] = (rgb,original,truth,region)
        predictions[int(rgb[0,0,0])] = pred
    manifest = {"config":cfg,"analysis_fingerprint":"a"*64,"inventory":inventory}
    rows = image_rows(evals,cfg)
    request = write_record(tmp/"fixture_request.json",make_request(manifest,rows,measurements))
    gt = {key(e): {"analysis_fingerprint":request["analysis_fingerprint"], **label_statistics(originals[key(e)][1],.01), "split":e["split"],"name":e["name"],
                   "image":image_identity(e),"label_sha256":e["label_sha256"]} for e in inventory}
    return cfg,cfg6,m6,request,gt,originals,predictions,measurements,evals


def main():
    tmp = Path(tempfile.mkdtemp(prefix="phase7_check_"))
    probe = tmp / "container_import"
    probe.mkdir()
    shutil.copy2(REPO / "modal_phase7.py", probe / "modal_phase7.py")
    shutil.copytree(REPO / "src", probe / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    startup = subprocess.run(
        [sys.executable, "-c", "import modal; modal.is_local = lambda: False; import modal_phase7; assert modal_phase7.image is None; assert isinstance(modal_phase7.survey_remote, modal.Function)"],
        cwd=probe, env={**os.environ, "PYTHONPATH": str(probe)}, capture_output=True, text=True,
    )
    check("container startup imports without the local Phase-6 app or scripts directory", startup.returncode == 0)
    if startup.returncode:
        print(startup.stderr, flush=True)
    cfg,cfg6,m6,request,gt,originals,predictions,measured,evals = fixture(tmp)
    check("independent config is outside the training glob and validates", load_config(REPO/"configs/phase7/phase7_v1.yaml")==DEFAULTS)
    check("unknown fields, bool counts, changed populations and unsafe roots reject", all((
        raises(lambda:validate_config({**DEFAULTS,"typo":1})),
        raises(lambda:validate_config({**DEFAULTS,"gallery":{**DEFAULTS["gallery"],"max_cases":True}})),
        raises(lambda:validate_config({**DEFAULTS,"splits":{}})),
        raises(lambda:validate_config({**DEFAULTS,"analysis_root":"/data/runs"})),
        raises(lambda:validate_config({**DEFAULTS,"analysis_name":"../runs"})),
        raises(lambda:validate_config({**DEFAULTS,"execution":{**DEFAULTS["execution"],"gpu_timeout_seconds":3600}})))))
    h1=np.array([[9,1],[0,0]]);h2=np.array([[0,0],[1,0]])
    check("IoU pooling differs from averaging image ratios", ious(h1+h2)[0]==9/11 and np.mean([ious(h1)[0],ious(h2)[0]])!=ious(h1+h2)[0])
    check("undefined scores stay distinct from zero, and empty summaries stay undefined", describe([None,0,1])=={
        "n_total":3,"n_valid":2,"n_undefined":1,"mean":.5,"median":.5,"q25":.25,"q75":.75} and describe([None])["mean"] is None)
    p = tmp/"duplicate.json";p.write_text('{"image":1,"image":2}')
    check("duplicate JSON image keys are rejected before a dict can hide them", raises(lambda:strict_json(p)))
    reversed_evals = copy.deepcopy(evals)
    for v in reversed_evals.values():v["per_image"] = dict(reversed(list(v["per_image"].items())))
    check("joins use split/original name, independent of listing order", image_rows(reversed_evals,cfg)==request["images"])
    bad = copy.deepcopy(evals);next(iter(bad.values()))["per_image"].pop(next(iter(next(iter(bad.values()))["per_image"])))
    check("missing images and resolution mismatches reject", raises(lambda:image_rows(bad,cfg)))
    bad = copy.deepcopy(evals);next(iter(next(iter(bad.values()))["per_image"].values()))["size"]=[61,40]
    check("different model mask coordinates reject", raises(lambda:image_rows(bad,cfg)))
    a = "youtube_Title-abcdefghijk-1080p-123"
    aliases = video_aliases([a+"_001",a+" (1)_002","youtube_Short_003","youtube_Short (1)_004"])
    check("YouTube source is separate from own; aliases require an embedded ID", source("youtube_x_1")=="youtube" and source("escape_x_1")=="own"
          and aliases[a+" (1)"]==a and aliases["youtube_Short (1)"]=="youtube_Short (1)")
    runs = cfg["baseline_runs"]+cfg["cascade_runs"]
    summaries = source_summaries(request["images"],runs,gt)
    yt = next(s for s in summaries if s["source"]=="youtube")["classes"]["interacting_object"]
    check("shared eligibility excludes undefined values without zero filling", yt["paired_four_run_defined"]["n_eligible"]==1 and yt["paired_four_run_defined"]["n_excluded"]==1)
    check("GT presence is model-independent and absent-image predictions remain visible", yt["gt_present"]["n_images"]==1 and yt["gt_absent"]["n_images"]==1
          and yt["gt_absent"]["prediction_present_count"][runs[0]]==1 and yt["gt_absent"]["prediction_present_count"][runs[2]]==0)
    wrong_rows=copy.deepcopy(request["images"])
    next(r for r in wrong_rows if r["name"].endswith("000011"))["scores"][runs[0]]["interacting_object"] = .4
    check("GT-absent nonzero saved IoU is rejected", raises(lambda:source_summaries(wrong_rows,runs,gt)))
    r=request["images"][0];v=r["scores"];p=r["paired"]["interacting_object"]
    check("paired delta averages same-seed deltas with equal image weight", p["mean_delta"]==np.mean([v[runs[2]]["interacting_object"]-v[runs[0]]["interacting_object"],v[runs[3]]["interacting_object"]-v[runs[1]]["interacting_object"]]))
    zero=label_statistics(np.zeros((10,10),np.uint8),.01)
    check("empty object geometry is completed, ignore is excluded, ID 5 is retained", zero["object_components"]==[] and zero["two_hand_pixels"]==0
          and gt[key(r)]["class_counts"]["ignore"]==9 and gt[key(r)]["two_hand_pixels"]==1)
    check("smallness uses original frame area and 8-connected GT components", any(c["area"]==1 and c["small"] for c in gt[key(r)]["object_components"])
          and any(not c["small"] for c in gt[key(r)]["object_components"]))
    tiny_gt=np.array([[0,1,2,255],[2,2,0,255]],np.uint8)
    tiny_p=np.array([[2,2,0,2],[2,1,0,2]],np.uint8)
    fp,fn=object_errors(tiny_p,tiny_gt)
    check("object FP includes hand confusion and excludes ignore; FN includes object-to-hand", int(fp.sum())==2 and int(fn.sum())==2 and not fp[:,3].any())
    check("misaligned masks and invalid prediction class IDs reject", raises(lambda:object_errors(tiny_p[:1],tiny_gt)) and raises(lambda:object_errors(tiny_p+3,tiny_gt)))
    shortlist=choose_shortlist(request,gt)
    reversed_req=copy.deepcopy(request);reversed_req["images"].reverse();reversed_req["clutter_hints"].reverse()
    # Hints have a declared deterministic ranking, not the caller's list order.
    reversed_req["clutter_hints"] = clutter_hints(list(reversed(measured)),cfg)
    check("shortlist is deterministic with source/video diversity", shortlist==choose_shortlist(reversed_req,dict(reversed(list(gt.items()))))
          and len({r["source"] for r in shortlist})==5 and max(__import__('collections').Counter((r["source"],r["diversity_video"]) for r in shortlist).values())<=2)
    check("file IDs distinguish names that sanitise/truncate to the same title", image_key({"split":"x","name":"a b"})!=image_key({"split":"x","name":"a_b"})
          and image_key({"split":"x","name":"same"})!=image_key({"split":"y","name":"same"}))
    hints=request["clutter_hints"]
    check("SAM event selection is strict and consistent at both seeds", {h["event"] for h in hints}=={"cascade_only","baseline_only"}
          and all(all(h["object_fractions"][r]>.5 for r in (cfg["cascade_runs"] if h["event"]=="cascade_only" else cfg["baseline_runs"])) for h in hints))
    equal=copy.deepcopy(measured[:1]);equal[0]["candidates"][0]["object_fractions"]={r:.5 for r in runs}
    check("fraction exactly one-half is not an FP event", clutter_hints(equal,cfg)==[])
    review=review_template(request,shortlist)
    check("appearance categories and captions are never auto-assigned", all(not c["tags"] and not c["caption"] and not c["selected"] for c in review["cases"]))
    for v in review["categories"].values():v.update(status="uncertain",note="Synthetic fixture: appearance not reviewed")
    for i,c in enumerate(review["cases"]):
        c.update(selected=True,preview=i<4,caption="Synthetic alignment and recovery fixture.")
    selection=write_record(tmp/"selection.json",build_selection(request,shortlist,review))
    bad=copy.deepcopy(review);bad["cases"][0]["tags"]=["transparent"]
    check("manual tags require evidence and selected captions", raises(lambda:build_selection(request,shortlist,bad)))
    bad=copy.deepcopy(review);bad["cases"][0]["preview"]=False
    check("exactly four previews are required", raises(lambda:build_selection(request,shortlist,bad)))
    bad=copy.deepcopy(selection);bad["cases"][0]["caption"]="Changed"
    check("case fingerprints catch stale/mutated manual selections", raises(lambda:validate_selection(bad,request)))
    bad=copy.deepcopy(selection);bad["categories"]["lighting"]["status"]="pending"
    check("full gallery rejects unresolved appearance categories", raises(lambda:validate_selection(bad,request,final=True)))
    bad=copy.deepcopy(selection);bad["cases"]=[c for c in bad["cases"] if c["source"]!="youtube"]
    check("final selection must retain source coverage", raises(lambda:validate_selection(bad,request,final=True)))
    tagcase=next(c for c in selection["cases"] if c["name"]==hints[0]["name"] and c["split"]==hints[0]["split"])
    tagcase=copy.deepcopy(tagcase);tagcase["candidate"]=hints[0]
    e=next(e for e in request["manifest"]["inventory"] if key(e)==key(tagcase))
    rgb,original,truth,region=originals[key(e)]
    m=next(m for m in measured if key(m)==key(e))
    mask,candidate=selected_region(tagcase,truth,e,cfg6,m6,m)
    check("selected SAM overlays reconstruct the eligible bitmap and preserve ID/area/distance", np.array_equal(mask,region) and candidate["mask_id"]==0 and candidate["area"]==20)
    check("proposal and eligible-mask checksum mismatches stop reuse", raises(lambda:selected_region(tagcase,truth,e,cfg6,m6,{**m,"eligible_masks_sha256":"bad"}))
          and raises(lambda:selected_region(tagcase,truth,e,cfg6,m6,{**m,"proposal_masks_sha256":"bad"})))
    check("missing SAM cache fails without regenerating proposals", raises(lambda:selected_region(tagcase,truth,e,cfg6,{"proposal_fingerprint":"missing"},m)))
    stats=check_predictions(tagcase,predictions[int(rgb[0,0,0])],truth,cfg,mask,candidate)
    badpred=copy.deepcopy(predictions[int(rgb[0,0,0])]);badpred[runs[0]][0,0]=2
    check("selected inference must reproduce saved per-image IoUs and candidate fractions", len(stats)==4 and raises(lambda:check_predictions(tagcase,badpred,truth,cfg)))
    badcandidate=copy.deepcopy(candidate);badcandidate["object_fractions"][runs[0]]=.25
    check("same IoUs cannot bypass candidate-fraction reconciliation", raises(lambda:check_predictions(tagcase,predictions[int(rgb[0,0,0])],truth,cfg,mask,badcandidate)))
    panels=panel_arrays(np.zeros((*tiny_gt.shape,3),np.uint8),tiny_gt,tiny_p,tiny_p)
    check("all six tiles share geometry and error colours identify exact pixels", len(panels)==6 and len({p.shape for p in panels})==1
          and tuple(panels[4][0,0])==FP and tuple(panels[4][0,2])==FN and tuple(panels[4][0,3])!=FP)
    cropped=panel_arrays(np.zeros((*tiny_gt.shape,3),np.uint8),tiny_gt,tiny_p,tiny_p,crop=[1,0,3,2])
    check("matched crops preserve the same original pixels in both error panels", cropped[4].shape==(2,2,3)
          and tuple(cropped[4][0,0])==FP and tuple(cropped[5][0,1])==FN)
    render_example={**tagcase,"crop_xyxy":[16,12,42,30]}
    files,_=render_case(render_example,rgb,truth,predictions[int(rgb[0,0,0])],cfg,mask)
    sample=tmp/"layout.png";sample.write_bytes(next(iter(files.values())))
    check("real panel renderer writes decodable full-frame and matched-crop PNGs", all(Image.open(io.BytesIO(v)).format=="PNG" for v in files.values()) and len(files)==2)
    malicious=copy.deepcopy(shortlist);malicious[0]["name"]="<script>alert(1)</script>"
    check("HTML captions/names are escaped", b"<script>alert" not in survey_index(malicious))
    fresh_manifest=copy.deepcopy(request["manifest"])
    fresh_manifest.update(config=copy.deepcopy(DEFAULTS),code_sha256=code_hashes())
    fresh=write_record(tmp/"fresh.json",make_request(fresh_manifest,request["images"],measured))
    frozen6={"config":{k:v for k,v in cfg6.items() if k!="execution"},"prepared_execution":cfg6["execution"]}
    with patch("src.phase7_workflow.load_inputs",return_value=(fresh_manifest,request["images"],measured,None,None)), \
         patch("src.phase7_workflow.read_record",return_value=frozen6),patch("src.phase7_workflow.load_manifest",return_value=m6):
        check("remote request reconstruction retains the frozen runtime validator", verify_request(fresh)[2]==m6)
        changed=copy.deepcopy(bare(fresh));changed["images"][0]["size"]=[1,1]
        changed=write_record(tmp/"changed.json",changed)
        check("resealed but altered derived request data is rejected", raises(lambda:verify_request(changed)))
        changed=copy.deepcopy(bare(fresh));changed["manifest"]["code_sha256"]={}
        changed=write_record(tmp/"changed_code.json",changed)
        check("changed Phase-7 source identity is rejected before remote work", raises(lambda:verify_request(changed)))
    workflow_checks(tmp,cfg,cfg6,m6,request,selection,gt,originals,predictions,measured)
    print(f"\n{sum(RESULTS.values())}/{len(RESULTS)} checks passed. Synthetic layout: {sample}")
    if not all(RESULTS.values()):raise SystemExit(1)


def workflow_checks(tmp,cfg,cfg6,m6,request,selection,gt,originals,predictions,measured):
    root=tmp/"remote_outputs"
    counts={"scan":0,"predict":0}
    def scan(e,c6,c):
        counts["scan"]+=1
        return label_statistics(originals[key(e)][1],.01)
    def original(e,c):
        rgb,_,truth,_=originals[key(e)]
        return rgb,truth
    verified=(cfg,cfg6,m6,measured)
    class Interrupted(Exception):pass
    commits=[]
    def interrupt():
        commits.append(1)
        if len(commits)==1:raise Interrupted
    with patch("src.phase7_workflow.verify_request",return_value=verified),patch("src.phase7_workflow.output_root",return_value=root), \
         patch("src.phase7_workflow.scan_label",side_effect=scan),patch("src.phase7_workflow.load_original",side_effect=original):
        check("CPU scan interruption leaves completed per-image records", raises(lambda:survey(request,interrupt),Interrupted) and counts["scan"]==3)
        survey(request)
        check("CPU scan resumes without rescanning completed labels and covers exact inventory", counts["scan"]==len(gt)
              and len(gt_lookup(read_record(root/"gt_stats.json"),request))==len(gt))
        survey(request)
        check("completed CPU survey is reusable including GT-absent records", counts["scan"]==len(gt))
        fetched=fetch_files(cfg)
        check("fetch allowlist excludes dense arrays, weights, proposal caches and scan records", "gt_stats.json" in fetched
              and not any(n.startswith(("survey_records/","thumbnail_records/")) or n.endswith((".pt",".npy",".gz")) for n in fetched))
    # The survey was built with the same rows/geometry; the execution setting does not alter cases.
    class Predictor:
        def __init__(self,*args):pass
        def __call__(self,rgb,truth):
            counts["predict"]+=1
            return predictions[int(rgb[0,0,0])]
    png=io.BytesIO();Image.new("RGB",(20,20)).save(png,format="PNG");content=png.getvalue()
    def render(c,*args):return {f"gallery/{c['case_id']}_full.png":content},None
    with patch("src.phase7_workflow.verify_request",return_value=verified),patch("src.phase7_workflow.output_root",return_value=root), \
         patch("src.phase7_workflow.load_original",side_effect=original),patch("src.clutter_workflow.SegmentationPredictor",Predictor), \
         patch("src.phase7_render.render_case",side_effect=render):
        commits.clear()
        check("selected rendering is interruptible after one committed image", raises(lambda:render_selected(request,selection,commit=interrupt),Interrupted) and counts["predict"]==1)
        render_selected(request,selection)
        check("preview resume computes only the missing cases", counts["predict"]==4 and read_record(root/"preview_summary.json")["n_cases"]==4)
        render_selected(request,selection)
        check("completed preview reuses checked panels without inference", counts["predict"]==4)
        check("full rendering requires explicit preview acceptance", raises(lambda:render_selected(request,selection,preview=False)))
        write_record(root/"case_selection.json",bare(selection))
        approve_preview(root)
        approval=read_record(root/"preview_approval.json");preview=read_record(root/"preview_summary.json")
        validate_approval(approval,preview,selection)
        render_selected(request,selection)
        validate_approval(approval,read_record(root/"preview_summary.json"),selection)
        check("resume timing alone does not invalidate reviewed visual content", counts["predict"]==4)
        wrong=copy.deepcopy(approval);wrong["preview_token"]="changed"
        check("preview acceptance is bound to exact completed files/cases", raises(lambda:validate_approval(wrong,preview,selection)))
        render_selected(request,selection,preview=False,approval=approval)
        check("full gallery reconciles exact selected coverage and reuses preview predictions", counts["predict"]==len(selection["cases"])
              and read_record(root/"gallery_summary.json")["n_cases"]==len(selection["cases"]))
        check("status distinguishes computed outputs and pending final user verification", status(cfg)["gallery_complete"] is True
              and status(cfg)["user_final_verification"]=="pending")
        files=fetch_files(cfg)
        check("final fetch contains only referenced gallery files and compact case records", "gallery/index.html" in files and "gallery/preview.html" in files
              and sum(n.startswith("case_records/") for n in files)==len(selection["cases"]))
        p=root/next(n for n in files if n.endswith("_full.png"));p.write_bytes(b"corrupt")
        check("fetch rejects corrupted panel bytes", raises(lambda:fetch_files(cfg)))
        render_selected(request,selection,preview=False,approval=approval)
        check("resume repairs only the damaged selected panel", counts["predict"]==len(selection["cases"])+1)
        limited=copy.deepcopy(cfg);limited["execution"]["fetch_max_bytes"]=1024
        check("fetch enforces the explicit byte ceiling", raises(lambda:fetch_files(limited)))
        saved=read_record(root/"gallery_summary.json")
        bad={**bare(saved),"files_sha256":{**saved["files_sha256"],"../escape.png":"bad"}}
        write_record(root/"gallery_summary.json",bad)
        check("allowlist refuses path traversal even in a checksummed summary", raises(lambda:fetch_files(cfg)))
        write_record(root/"gallery_summary.json",bare(saved))


if __name__=="__main__":main()
