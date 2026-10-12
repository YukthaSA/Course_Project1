"""
PURE Dataset Loader & Ingestion Module
Downloads the official PURE (PUblic REquirements) dataset from ISTI-CNR / Zenodo,
extracts the 18 structured XML requirements documents, and generates standardized SRS documents and a master catalog.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import zipfile
import io
import re
import requests
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PURELoader")

CNR_ZIP_URL = "http://nlreqdataset.isti.cnr.it/new/XMLZIPFile_02.zip"
ZENODO_URL = "https://zenodo.org/api/records/1414117/files/requirements-xml.zip/content"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

PURE_METADATA_MAP = {
    "cctns": {
        "title": "Crime and Criminal Tracking Network and Systems (CCTNS)",
        "domain": "Law Enforcement & Public Safety",
        "description": "Comprehensive requirements for police criminal investigation records, complaint management, biometric search, and citizen services portal."
    },
    "keepass": {
        "title": "KeePass Password Safe Specification",
        "domain": "Security, Authentication & Cryptography",
        "description": "Secure master key authentication, AES-256 encryption, password generation, clipboard protection, and encrypted database management."
    },
    "peppol": {
        "title": "PEPPOL Pan-European Public Procurement On-Line",
        "domain": "E-Commerce & Digital Procurement",
        "description": "Cross-border electronic ordering, invoice validation, secure transmission gateway, and multi-currency billing reconciliation."
    },
    "themas": {
        "title": "THEMAS Radiation Therapy Monitoring System",
        "domain": "Healthcare & Medical Devices",
        "description": "Safety-critical radiation therapy verification, oncology patient positioning, beam dosage calculation, and emergency halt protocols."
    },
    "gemini": {
        "title": "Gemini 8m Telescopes Control System",
        "domain": "Aerospace & Astronomical Observatory",
        "description": "Real-time telescope mount positioning, instrumentation control, sensor telemetry, and safety interlocks for twin 8m optical telescopes."
    },
    "gamma": {
        "title": "Gamma Ray Burst Detector (GROND / Gamma-J)",
        "domain": "Scientific Instrumentation & Astronomy",
        "description": "Automated trigger capture, multi-channel optical sensor calibration, and rapid burst event data transmission."
    },
    "ertms": {
        "title": "European Rail Traffic Management System (ERTMS)",
        "domain": "Railway Transportation & Safety-Critical",
        "description": "Standardized interoperable train control, automated speed supervision, emergency braking, and cab signaling."
    },
    "eirene": {
        "title": "EIRENE Railway Telecommunications System",
        "domain": "Railway Transportation & Telecommunications",
        "description": "GSM-R railway wireless radio network, emergency voice priority calling, and wayside-to-train data communication."
    },
    "tcs": {
        "title": "Intelligent Traffic Control System (TCS)",
        "domain": "Intelligent Transportation Systems",
        "description": "Adaptive urban traffic light optimization, intersection queue sensor feedback, and emergency vehicle green-wave preemption."
    },
    "phin": {
        "title": "Public Health Information Network (CDC PHIN)",
        "domain": "Public Health & Epidemiology",
        "description": "National disease surveillance notification, laboratory result messaging, vaccine registry exchange, and outbreak alerting."
    },
    "microcare": {
        "title": "Microcare Patient Tele-Monitoring Platform",
        "domain": "Healthcare & Telemedicine",
        "description": "Remote patient vital signs telemetry, medication reminder scheduling, and caregiver escalation paging."
    },
    "qheadache": {
        "title": "Quality Headache Medical Diagnostic Assistant",
        "domain": "Healthcare & Clinical Decision Support",
        "description": "Symptom questionnaire workflow, ICD-10 diagnostic differential reasoning, and clinical contraindication checks."
    },
    "video_search": {
        "title": "Content-Based Video Retrieval & Search Engine",
        "domain": "Multimedia & AI Information Retrieval",
        "description": "Video keyframe indexing, semantic tag search, subtitle full-text query, and video streaming playback."
    },
    "peering": {
        "title": "BGP Internet Route Peering Coordinator",
        "domain": "Internet Infrastructure & Networking",
        "description": "Border Gateway Protocol session negotiation, autonomous system route validation, and prefix filtering."
    },
    "dii": {
        "title": "Defense Information Infrastructure (DII COE)",
        "domain": "Defense & Secure Infrastructure",
        "description": "Mission-critical defense computing environment, role-based access control, cryptographic verification, and audit logging."
    },
    "blitdraft": {
        "title": "BlitDraft Enterprise Document System",
        "domain": "Enterprise Document Management",
        "description": "Multi-user revision versioning, vector rendering, electronic signatures, and access control."
    },
    "get_real": {
        "title": "Get Real Low-Latency Audio Processing Engine",
        "domain": "Multimedia & Audio Signal Processing",
        "description": "Real-time digital signal processing, spatial audio filtering, buffer overrun prevention, and hardware driver abstraction."
    }
}

class PUREDatasetLoader:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or r"C:\Users\HP\Course_Project1"
        self.data_dir = os.path.join(self.base_dir, "data", "pure")
        self.xml_dir = os.path.join(self.data_dir, "xml")
        self.samples_pure_dir = os.path.join(self.base_dir, "samples", "pure")
        self.catalog_path = os.path.join(self.samples_pure_dir, "catalog.json")

        os.makedirs(self.xml_dir, exist_ok=True)
        os.makedirs(self.samples_pure_dir, exist_ok=True)

    def download_dataset(self, force: bool = False) -> bool:
        """Download PURE XML documents package from CNR mirror or Zenodo"""
        existing_xmls = [f for f in os.listdir(self.xml_dir) if f.endswith(".xml")]
        if existing_xmls and not force:
            logger.info(f"PURE XML dataset already present at '{self.xml_dir}' ({len(existing_xmls)} XML files).")
            return True

        urls_to_try = [
            CNR_ZIP_URL,
            ZENODO_URL
        ]

        zip_data = None
        for url in urls_to_try:
            logger.info(f"Attempting to download PURE dataset from {url}...")
            try:
                r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
                if r.status_code == 200 and len(r.content) > 10000:
                    logger.info(f"Successfully downloaded {len(r.content):,} bytes from {url}")
                    zip_data = r.content
                    break
                else:
                    logger.warning(f"Download returned status {r.status_code} ({len(r.content)} bytes)")
            except Exception as e:
                logger.warning(f"Error fetching from {url}: {e}")

        if not zip_data:
            logger.error("Could not download PURE dataset package from any source.")
            return False

        # Extract XML files
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                extracted_count = 0
                for member in z.namelist():
                    if member.endswith(".xml") and not member.startswith("__MACOSX"):
                        filename = os.path.basename(member)
                        if filename:
                            target_file = os.path.join(self.xml_dir, filename)
                            with open(target_file, "wb") as f_out:
                                f_out.write(z.read(member))
                            extracted_count += 1
                logger.info(f"Extracted {extracted_count} XML documents to {self.xml_dir}")
                return True
        except Exception as e:
            logger.error(f"Error extracting zip: {e}")
            return False

    def parse_all_documents(self) -> List[Dict[str, Any]]:
        """Parse all XML documents into clean SRS text and index in catalog.json"""
        xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith(".xml")]
        if not xml_files:
            if not self.download_dataset():
                return []
            xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith(".xml")]

        catalog = []
        logger.info(f"Parsing {len(xml_files)} PURE XML documents...")

        for filename in sorted(xml_files):
            xml_path = os.path.join(self.xml_dir, filename)
            doc_data = self._parse_single_xml(xml_path, filename)
            if doc_data:
                catalog.append(doc_data)

        # Write catalog.json
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

        # Also write a helpful README in samples/pure/
        readme_path = os.path.join(self.samples_pure_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# PURE (PUblic REquirements) Dataset Collection\n\n")
            f.write("This directory contains real-world Software Requirements Specifications (SRS) ")
            f.write("derived from the official PURE benchmark dataset (Ferrari et al., IEEE RE 2017).\n\n")
            f.write(f"Total Ingested SRS Documents: **{len(catalog)}**\n\n")
            f.write("| ID | Document Title | Domain | Requirements Count |\n")
            f.write("|---|---|---|---|\n")
            for d in catalog:
                f.write(f"| `{d['id']}` | {d['title']} | {d['domain']} | {d['requirements_count']} |\n")

        logger.info(f"Catalog saved to '{self.catalog_path}' with {len(catalog)} documents.")
        return catalog

    def _parse_single_xml(self, xml_path: str, filename: str) -> Optional[Dict[str, Any]]:
        try:
            with open(xml_path, "rb") as f:
                content = f.read()
            tree = ET.fromstring(content)
        except Exception as e:
            logger.error(f"Error parsing XML file {filename}: {e}")
            return None

        # Clean document key identifier
        raw_key = os.path.splitext(filename)[0].lower()
        clean_id = re.sub(r"[^a-z0-9]+", "_", raw_key).strip("_")

        # Determine metadata
        meta = None
        for k, v in PURE_METADATA_MAP.items():
            if k in clean_id:
                meta = v
                break

        # Fallback metadata from XML title tags
        if not meta:
            title_nodes = tree.findall(".//{req_document.xsd}title")
            doc_title = " ".join("".join(t.itertext()).strip() for t in title_nodes[:2] if "".join(t.itertext()).strip())
            if not doc_title:
                doc_title = clean_id.replace("_", " ").title()
            meta = {
                "title": doc_title,
                "domain": "Systems & Software Engineering",
                "description": f"Public Software Requirements Specification from the PURE benchmark dataset ({clean_id})."
            }

        # Extract requirements
        extracted_reqs = []
        
        # 1. Look for explicit <req> elements
        req_nodes = tree.findall(".//{req_document.xsd}req")
        req_counter = 1
        for node in req_nodes:
            text = "".join(node.itertext()).strip()
            # Clean up whitespace and control chars
            text = re.sub(r"\s+", " ", text)
            if len(text) > 12:
                req_id = node.attrib.get("id") or f"REQ-{req_counter:03d}"
                if not req_id.startswith("REQ-"):
                    req_id = f"REQ-{req_counter:03d}"
                req_counter += 1

                # Classify category
                text_lower = text.lower()
                category = "Functional"
                if any(w in text_lower for w in ["security", "authenticate", "encrypt", "password", "tls", "https", "permission", "access control", "hash", "breach"]):
                    category = "Security"
                elif any(w in text_lower for w in ["performance", "throughput", "latency", "response time", "seconds", "ms", "real-time"]):
                    category = "Performance"
                elif any(w in text_lower for w in ["ui", "display", "screen", "button", "dialog", "view", "interface", "menu", "page"]):
                    category = "UI"
                elif any(w in text_lower for w in ["safety", "fail-safe", "brake", "hazard", "interlock", "emergency", "radiation"]):
                    category = "Safety"

                priority = "High" if category in ["Security", "Safety"] or any(w in text_lower for w in ["shall", "must", "critical"]) else "Medium"
                
                extracted_reqs.append({
                    "id": req_id,
                    "text": text,
                    "category": category,
                    "priority": priority
                })

        # 2. If <req> nodes were not present or few, search text_body / p / item
        if len(extracted_reqs) < 5:
            body_nodes = (
                tree.findall(".//{req_document.xsd}item") + 
                tree.findall(".//{req_document.xsd}text_body") + 
                tree.findall(".//{req_document.xsd}p")
            )
            for node in body_nodes:
                text = "".join(node.itertext()).strip()
                text = re.sub(r"\s+", " ", text)
                if len(text) > 20 and any(kw in text.lower() for kw in ["shall", "must", "will", "should", "website", "system", "user", "can", "allow", "support", "provide", "display"]):
                    req_id = f"REQ-{req_counter:03d}"
                    req_counter += 1
                    category = "Functional"
                    if any(w in text.lower() for w in ["security", "authenticate", "login", "password"]):
                        category = "Security"
                    elif any(w in text.lower() for w in ["ui", "website", "page", "display", "link"]):
                        category = "UI"
                    extracted_reqs.append({
                        "id": req_id,
                        "text": text,
                        "category": category,
                        "priority": "High" if "shall" in text.lower() or category == "Security" else "Medium"
                    })

        # Deduplicate and sort
        seen = set()
        unique_reqs = []
        for r in extracted_reqs:
            normalized = r["text"].lower()[:80]
            if normalized not in seen:
                seen.add(normalized)
                unique_reqs.append(r)

        # Generate readable SRS document
        srs_filename = f"{clean_id}_srs.txt"
        srs_path = os.path.join(self.samples_pure_dir, srs_filename)

        with open(srs_path, "w", encoding="utf-8") as f_out:
            f_out.write("=" * 75 + "\n")
            f_out.write("SYSTEM REQUIREMENTS SPECIFICATION (SRS)\n")
            f_out.write(f"PROJECT: {meta['title'].upper()}\n")
            f_out.write(f"DOMAIN:  {meta['domain']}\n")
            f_out.write(f"SOURCE:  PURE Benchmark Dataset (IEEE RE 2017 / ISTI-CNR)\n")
            f_out.write("=" * 75 + "\n\n")

            f_out.write("1. SYSTEM OVERVIEW & SCOPE\n")
            f_out.write(f"   {meta['description']}\n\n")

            f_out.write(f"2. TESTABLE SOFTWARE REQUIREMENTS ({len(unique_reqs)} Specifications)\n")
            f_out.write("-" * 75 + "\n\n")

            for req in unique_reqs:
                f_out.write(f"[{req['id']}] [{req['category']}] Priority: {req['priority']}\n")
                f_out.write(f"Requirement: {req['text']}\n\n")

        doc_summary = {
            "id": clean_id,
            "filename": srs_filename,
            "xml_filename": filename,
            "title": meta["title"],
            "domain": meta["domain"],
            "description": meta["description"],
            "path": srs_path,
            "relative_path": os.path.relpath(srs_path, self.base_dir),
            "requirements_count": len(unique_reqs),
            "categories": sorted(list(set(r["category"] for r in unique_reqs))) if unique_reqs else ["Functional"]
        }
        return doc_summary

    def get_catalog(self) -> List[Dict[str, Any]]:
        """Return the list of all available PURE documents"""
        if os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self.parse_all_documents()

    def load_document(self, doc_id_or_name: str) -> Optional[str]:
        """Load text content of a PURE document by ID, title, or filename"""
        catalog = self.get_catalog()
        query = doc_id_or_name.lower().strip()
        for doc in catalog:
            if (query == doc["id"].lower() or 
                query in doc["filename"].lower() or 
                query in doc["title"].lower() or
                query in doc["domain"].lower()):
                if os.path.exists(doc["path"]):
                    with open(doc["path"], "r", encoding="utf-8") as f:
                        return f.read()
        return None

if __name__ == "__main__":
    loader = PUREDatasetLoader()
    print("[PURE Dataset Loader] Starting ingestion...")
    catalog = loader.parse_all_documents()
    print(f"\n[OK] Ingested {len(catalog)} real SRS documents from PURE Dataset:")
    for d in catalog:
        print(f"  * [{d['id']}] {d['title']} ({d['domain']}) -> {d['requirements_count']} requirements")
