# ===================================================================
# FILE: app/services/kb_service.py
# ===================================================================

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
from pathlib import Path
from rapidfuzz import process, fuzz

from app.core.config import settings
from app.core.exceptions import KnowledgeBaseError

logger = logging.getLogger(__name__)

# ======================= DATA STRUCTURES ============================

@dataclass
class DosenInfo:
    nama_lengkap: str = "-"
    panggilan: str = "-"
    nidn: str = "-"
    no_hp: str = "-"
    matakuliah: str = "-"
    semester: int = 0
    prodi: str = "-"
    alias: Optional[Dict[str, List[str]]] = None

@dataclass
class MataKuliahInfo:
    kode: str = "-"
    sks: int = 0
    semester: int = 0
    prodi: str = "-"
    prasyarat: str = "-"
    deskripsi: str = "-"
    kompetensi: str = "-"
    alias: Optional[Dict[str, List[str]]] = None

@dataclass
class JadwalInfo:
    mata_kuliah: str = "-"
    kode: str = "-"
    sks: int = 0
    hari: str = "-"
    jam: str = "-"
    jam_mulai: float = 0.0
    jam_selesai: float = 0.0
    ruang: str = "-"
    kelas: str = "-"
    semester: int = 0
    prodi: str = "-"
    alias: Optional[Dict[str, List[str]]] = None

@dataclass
class KalenderInfo:
    tahun: str
    semester: str
    kegiatan: str
    mulai: str
    selesai: str
    target: str
    keterangan: str

@dataclass
class SkripsiInfo:
    prodi: str
    sks_minimum: int
    semester_minimum: int
    ipk_minimum: float
    matkul_wajib: str
    dokumen: str
    prosedur: str

@dataclass
class RegulasiBatasSKS:
    semester: str
    ipk_minimum: float
    ipk_maksimum: float
    sks_maksimal: int
    sks_minimal: int
    prodi: str
    keterangan: str

# ======================= KNOWLEDGE BASE LOADER =====================

class KnowledgeBaseLoader:
    def __init__(self) -> None:
        self.kb_file_path: Path = settings.KB_PATH
        self.templates_path: Path = settings.TEMPLATES_PATH

        self.dosen_data: Dict[str, DosenInfo] = {}
        self.matakuliah_data: Dict[str, MataKuliahInfo] = {}
        self.jadwal_data: List[JadwalInfo] = []
        self.kalender_data: List[KalenderInfo] = []
        self.skripsi_data: List[SkripsiInfo] = []
        self.regulasi_sks_data: List[RegulasiBatasSKS] = []

        self.response_templates: Dict[str, Dict[str, Any]] = {}

        self.dosen_by_matkul: Dict[str, List[str]] = defaultdict(list)
        self.matkul_by_prodi: Dict[str, List[str]] = defaultdict(list)
        self.jadwal_by_hari: Dict[str, List[JadwalInfo]] = defaultdict(list)
        self.jadwal_by_matkul: Dict[str, List[JadwalInfo]] = defaultdict(list)
        self.regulasi_by_prodi: Dict[str, List[RegulasiBatasSKS]] = defaultdict(list)

        self.is_loaded: bool = False
        self.load_time: float = 0.0
        self.raw_kb_data: Dict[str, Any] = {}

    def load_knowledge_base(self) -> bool:
        try:
            start_time = time.time()
            logger.info("🚀 Memulai proses loading knowledge base...")

            if not self.kb_file_path.exists():
                raise FileNotFoundError(f"KB file not found: {self.kb_file_path}")

            with open(self.kb_file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, dict):
                raise KnowledgeBaseError("KB harus berupa dictionary level atas")

            self.raw_kb_data = raw_data or {}
            self._parse_all_data(raw_data)

            if self.templates_path.exists():
                with open(self.templates_path, "r", encoding="utf-8") as f:
                    self.response_templates = json.load(f)
                logger.info(f"📝 Loaded {len(self.response_templates)} response templates")

            self.load_time = time.time() - start_time
            self.is_loaded = True
            logger.info(f"✅ Knowledge base berhasil dimuat dalam {self.load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading knowledge base: {str(e)}")
            self.is_loaded = False
            return False

    def _safe_dataclass_init(self, dataclass_type, data: Dict[str, Any]):
        field_names = {f.name for f in dataclass_type.__dataclass_fields__.values()}
        safe_data = {field: data.get(field, getattr(dataclass_type, field, None)) for field in field_names}
        return dataclass_type(**safe_data)

    def _parse_all_data(self, raw: Dict[str, Any]):
        # ---------- Dosen ----------
        if isinstance(raw.get("dosen"), dict):
            for _, v in raw["dosen"].items():
                try:
                    if not v.get("nama_lengkap"):
                        continue
                    info = self._safe_dataclass_init(DosenInfo, v)
                    main_key = info.nama_lengkap.lower()
                    self.dosen_data[main_key] = info
                    self.dosen_by_matkul[info.matakuliah.lower()].append(main_key)
                    if info.alias and "nama_lengkap" in info.alias:
                        for alias in info.alias["nama_lengkap"]:
                            alias_key = alias.lower()
                            self.dosen_data[alias_key] = info
                            self.dosen_by_matkul[info.matakuliah.lower()].append(alias_key)
                except Exception as e:
                    logger.warning(f"⚠️ Skip dosen entri invalid: {v} | {e}")

        # ---------- Mata Kuliah ----------
        if isinstance(raw.get("mata_kuliah"), dict):
            for key, v in raw["mata_kuliah"].items():
                try:
                    info = self._safe_dataclass_init(MataKuliahInfo, v)
                    main_key = key.lower()
                    self.matakuliah_data[main_key] = info
                    self.matkul_by_prodi[info.prodi.lower()].append(main_key)
                    if info.alias and "mata_kuliah" in info.alias:
                        for alias in info.alias["mata_kuliah"]:
                            alias_key = alias.lower()
                            self.matakuliah_data[alias_key] = info
                            self.matkul_by_prodi[info.prodi.lower()].append(alias_key)
                except Exception as e:
                    logger.warning(f"⚠️ Skip matakuliah entri invalid: {v} | {e}")

        # ---------- Jadwal ----------
        if isinstance(raw.get("jadwal"), list):
            for v in raw["jadwal"]:
                try:
                    info = self._safe_dataclass_init(JadwalInfo, v)
                    self.jadwal_data.append(info)
                    mk_key = info.mata_kuliah.lower()
                    self.jadwal_by_hari[info.hari.lower()].append(info)
                    self.jadwal_by_matkul[mk_key].append(info)
                    if info.alias and "mata_kuliah" in info.alias:
                        for alias in info.alias["mata_kuliah"]:
                            alias_key = alias.lower()
                            self.jadwal_by_matkul[alias_key].append(info)
                except Exception as e:
                    logger.warning(f"⚠️ Skip jadwal entri invalid: {v} | {e}")

        # ---------- Kalender ----------
        if isinstance(raw.get("kalender"), list):
            for v in raw["kalender"]:
                try:
                    self.kalender_data.append(KalenderInfo(**v))
                except TypeError as e:
                    logger.warning(f"⚠️ Skip kalender entri invalid: {v} | {e}")

        # ---------- Skripsi ----------
        if isinstance(raw.get("skripsi"), list):
            for v in raw["skripsi"]:
                try:
                    self.skripsi_data.append(SkripsiInfo(**v))
                except TypeError as e:
                    logger.warning(f"⚠️ Skip skripsi entri invalid: {v} | {e}")

        # ---------- Regulasi SKS ----------
        if isinstance(raw.get("regulasi_sks"), list):
            for v in raw["regulasi_sks"]:
                try:
                    info = RegulasiBatasSKS(**v)
                    self.regulasi_sks_data.append(info)
                    self.regulasi_by_prodi[info.prodi.lower()].append(info)
                except TypeError as e:
                    logger.warning(f"⚠️ Skip regulasi_sks entri invalid: {v} | {e}")

# ======================= FUZZY MATCHING UTILS =======================

def _fuzzy_lookup(query: str, choices: List[str], threshold: int = 80) -> Optional[str]:
    if not query or not choices:
        return None
    result = process.extractOne(query.lower().strip(), choices, scorer=fuzz.ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None

# ======================= QUERY INTERFACE ===========================

class KnowledgeBaseQuery:
    def __init__(self, kb_loader: KnowledgeBaseLoader):
        self.kb = kb_loader
        if not self.kb.is_loaded:
            raise KnowledgeBaseError("Knowledge base not loaded. Call initialize_knowledge_base() first.")

    def _normalize_prodi(self, prodi: str) -> str:
        if not prodi:
            return "teknik informatika"
        prodi = prodi.lower().strip()
        if prodi in ["ti", "it"]:
            return "teknik informatika"
        if prodi in ["si", "sistem informasi"]:
            return "sistem informasi"
        return prodi

    def find_dosen_by_matkul(self, mata_kuliah: str) -> List[DosenInfo]:
        matkul_key = _fuzzy_lookup(mata_kuliah, list(self.kb.dosen_by_matkul.keys())) or mata_kuliah.lower().strip()
        return [self.kb.dosen_data.get(n) for n in self.kb.dosen_by_matkul.get(matkul_key, []) if n in self.kb.dosen_data]

    def find_jadwal_by_matkul(self, mata_kuliah: str) -> List[JadwalInfo]:
        matkul_key = _fuzzy_lookup(mata_kuliah, list(self.kb.jadwal_by_matkul.keys())) or mata_kuliah.lower().strip()
        return self.kb.jadwal_by_matkul.get(matkul_key, [])

    def find_jadwal_by_hari(self, hari: str) -> List[JadwalInfo]:
        hari_key = _fuzzy_lookup(hari, list(self.kb.jadwal_by_hari.keys()), threshold=75) or hari.lower().strip()
        return self.kb.jadwal_by_hari.get(hari_key, [])

    def get_matakuliah_detail(self, mata_kuliah: str) -> Optional[MataKuliahInfo]:
        matkul_key = _fuzzy_lookup(mata_kuliah, list(self.kb.matakuliah_data.keys())) or mata_kuliah.lower().strip()
        return self.kb.matakuliah_data.get(matkul_key)

    def get_kalender_akademik(self, prodi: str, semester: Optional[str] = None) -> Optional[KalenderInfo]:
        prodi = self._normalize_prodi(prodi)
        return next(
            (k for k in self.kb.kalender_data if prodi in k.keterangan.lower() and (not semester or semester in k.semester.lower())),
            None,
        )

    def get_jadwal_semester(self, semester: str) -> Optional[KalenderInfo]:
        return next((k for k in self.kb.kalender_data if semester.lower() in k.semester.lower()), None)

    def get_dosen_pengampu(self, mata_kuliah: str) -> Dict[str, str]:
        dosen_list = self.find_dosen_by_matkul(mata_kuliah)
        if dosen_list:
            d = dosen_list[0]
            return {"dosen": d.nama_lengkap, "semester": d.semester, "prodi": d.prodi}
        return {}

    def get_jadwal_by_matakuliah(self, mata_kuliah: str) -> Dict[str, str]:
        jadwal_list = self.find_jadwal_by_matkul(mata_kuliah)
        if jadwal_list:
            j = jadwal_list[0]
            return {"hari": j.hari, "jam": j.jam, "ruang": j.ruang}
        return {}

    def get_dosen_info(self, dosen_name: str) -> Dict[str, str]:
        normalized_name = dosen_name.lower().replace("dosen", "").replace("pak", "").replace("bu", "").strip()
        dosen_key = _fuzzy_lookup(normalized_name, list(self.kb.dosen_data.keys())) or normalized_name
        d = self.kb.dosen_data.get(dosen_key)
        if d:
            return {
                "nama_lengkap": d.nama_lengkap,
                "no_hp": d.no_hp,
                "nidn": d.nidn,
                "panggilan": d.panggilan,
                "matakuliah": d.matakuliah,
                "semester": d.semester,
                "prodi": d.prodi,
            }
        return {}

    def get_batas_sks(self, semester: str, ipk: float, prodi: str) -> Optional[RegulasiBatasSKS]:
        prodi = self._normalize_prodi(prodi)
        return next((r for r in self.kb.regulasi_by_prodi.get(prodi, []) if r.semester == semester and r.ipk_minimum <= ipk <= r.ipk_maksimum), None)

    def get_syarat_skripsi(self, prodi: str, ipk: Optional[float] = None) -> List[SkripsiInfo]:
        prodi = self._normalize_prodi(prodi)
        return [s for s in self.kb.skripsi_data if s.prodi.lower().strip() == prodi and (ipk is None or ipk >= s.ipk_minimum)]

    def get_response_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        return self.kb.response_templates.get(template_name)

# ======================= GLOBAL INSTANCE ===========================

kb_loader = KnowledgeBaseLoader()
kb_query: Optional[KnowledgeBaseQuery] = None

def initialize_knowledge_base() -> bool:
    global kb_query
    try:
        if kb_loader.load_knowledge_base():
            kb_query = KnowledgeBaseQuery(kb_loader)
            logger.info("🎉 Global knowledge base initialized successfully")
            return True
        logger.error("❌ Failed to initialize knowledge base")
        return False
    except Exception as e:
        logger.error(f"❌ Error initializing knowledge base: {str(e)}")
        return False

def get_kb_query() -> KnowledgeBaseQuery:
    if kb_query is None:
        raise KnowledgeBaseError("Knowledge base not initialized. Call initialize_knowledge_base() first.")
    return kb_query

def is_kb_ready() -> bool:
    return kb_query is not None and kb_loader.is_loaded
