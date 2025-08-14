# ===================================================================
# FILE: app/services/template_service.py
# ===================================================================

import logging
from typing import Dict, Any, List, Optional
from rapidfuzz import process, fuzz

from app.services.kb_service import get_kb_query

logger = logging.getLogger(__name__)

# ===================== ENTITY → KB FIELD MAPPING ====================
ENTITY_KEY_MAPPING: Dict[str, str] = {
    "MATA_KULIAH": "mata_kuliah",
    "DOSEN": "nama_lengkap",
    "PRODI": "prodi",
    "SEMESTER": "semester",
    "HARI": "hari",
    "WAKTU": "jam",
    "RUANGAN": "ruang",
    "KELAS": "kelas",
    "SKS": "sks",
    "KODE_MATAKULIAH": "kode",
    "PRASYARAT": "prasyarat",
    "NAMA_LENGKAP": "nama_lengkap",
    "NAMA_PANGGILAN": "panggilan",
    "NIDN": "nidn",
    "PHONE": "no_hp",
    "IPK": "ipk_minimum",
    "TOTAL_SKS": "sks_minimum",
    "KEGIATAN": "kegiatan",
    "TANGGAL_MULAI": "mulai",
    "TANGGAL_SELESAI": "selesai",
    "DOKUMEN": "dokumen",
    "PROSEDUR": "prosedur",
}

# ======================== TEMPLATE SERVICE ==========================
class TemplateService:
    def __init__(self):
        self.kb_service = get_kb_query()
        self.templates: Dict[str, Dict[str, Any]] = self.kb_service.kb.response_templates
        logger.info(f"✅ TemplateService initialized successfully | {len(self.templates)} templates loaded")

    # ========================= NORMALISASI ===========================
    def _normalize_entity_value(self, value: Any) -> str:
        """Normalisasi entitas → string (list-safe)"""
        if isinstance(value, list) and len(value) > 0:
            return str(value[0]).strip()
        if isinstance(value, list):
            return ""
        return str(value or "").strip()

    def _normalize_dosen_name(self, name: str) -> str:
        if not name:
            return ""
        return name.lower().replace("dosen", "").replace("pak", "").replace("bu", "").strip()

    def _normalize_alias_query(self, value: str, intent: str, placeholder: str) -> str:
        """Sinkron alias → fuzzy match ke KB"""
        if not value:
            return value
        try:
            kb = self.kb_service.kb
            all_keys = []
            if intent in ["INFO_MATAKULIAH", "SKS_MATKUL", "PRASYARAT_MATKUL",
                          "JADWAL_MATAKULIAH", "JADWAL_HARI", "JADWAL_RUANGAN"]:
                all_keys = list(kb.matakuliah_data.keys())
            elif intent in ["DOSEN_PENGAMPU", "INFO_DOSEN_UMUM", "KONTAK_DOSEN", "NIDN_DOSEN"]:
                all_keys = list(kb.dosen_data.keys())
            elif intent in ["JADWAL_PRODI", "JADWAL_SEMESTER"]:
                return value  # prodi & semester tidak perlu fuzzy alias langsung

            if not all_keys:
                return value

            result = process.extractOne(value.lower().strip(), all_keys, scorer=fuzz.ratio)
            return result[0] if result and result[1] >= 75 else value
        except Exception as e:
            logger.warning(f"[ALIAS NORMALIZE WARNING] {e}")
            return value

    # ========================= MAIN ENTRY ============================
    def fill_template(
        self, intent: str, entities: Dict[str, Any],
        search_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        try:
            base_template = self._get_base_template(intent)
            return self._fill_kb_placeholders(intent, entities, base_template, search_results)
        except Exception as e:
            logger.error(f"[TEMPLATE ERROR] {e}")
            return f"Maaf, terjadi kesalahan saat memproses jawaban untuk intent '{intent}'."

    def _get_base_template(self, intent: str) -> str:
        template_data = self.templates.get(intent)
        return (
            template_data.get("template")
            if isinstance(template_data, dict)
            else str(template_data or "Maaf, saya belum memiliki template jawaban untuk intent ini.")
        )

    # ===================== FILL PLACEHOLDERS KB ======================
    def _fill_kb_placeholders(
        self, intent: str, entities: Dict[str, Any],
        template: str, search_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        try:
            for placeholder in ENTITY_KEY_MAPPING.keys():
                if "{" + placeholder + "}" not in template:
                    continue
                ent_value = self._normalize_entity_value(entities.get(placeholder))
                final_value = ent_value or self._lookup_from_kb(intent, placeholder, entities)
                template = template.replace("{" + placeholder + "}", str(final_value or "-"))

            if intent in ["OUT_OF_SCOPE", "PANDUAN_KRS", "PROSEDUR_CUTI"] and search_results:
                if search_results:
                    top_source = search_results[0].get("metadata", {}).get("source", "Referensi KB")
                    template = template.replace("{SOURCE}", top_source)
            return template
        except Exception as e:
            logger.error(f"[TEMPLATE KB LOOKUP ERROR] {e}")
            return template

    # ========================== KB LOOKUP ============================
    def _lookup_from_kb(self, intent: str, placeholder: str, entities: Dict[str, Any]) -> Optional[str]:
        try:
            # ===== BATAS SKS =====
            if intent == "BATAS_SKS" and placeholder == "SKS":
                prodi = self._normalize_entity_value(entities.get("PRODI")) or "Teknik Informatika"
                semester = self._normalize_entity_value(entities.get("SEMESTER")) or "1"
                ipk = float(self._normalize_entity_value(entities.get("IPK")) or 0.0)
                regulasi = self.kb_service.get_batas_sks(semester, ipk, prodi)
                return str(regulasi.sks_maksimal) if regulasi else "-"

            # ===== DOSEN PENGAMPU =====
            if intent == "DOSEN_PENGAMPU":
                matkul = self._normalize_alias_query(
                    self._normalize_entity_value(entities.get("MATA_KULIAH")), intent, placeholder
                )
                dosen = self.kb_service.get_dosen_pengampu(matkul)
                key_map = {
                    "DOSEN": "dosen",
                    "SEMESTER": "semester",
                    "PRODI": "prodi"
                }
                return str(dosen.get(key_map.get(placeholder, ""), "-"))

            # ===== INFO DOSEN UMUM =====
            if intent == "INFO_DOSEN_UMUM":
                dosen_name = self._normalize_alias_query(
                    self._normalize_dosen_name(self._normalize_entity_value(entities.get("DOSEN"))), intent, placeholder
                )
                dosen_info = self.kb_service.get_dosen_info(dosen_name)
                key_map = {
                    "NAMA_LENGKAP": "nama_lengkap",
                    "NAMA_PANGGILAN": "panggilan",
                    "PRODI": "prodi",
                    "MATA_KULIAH": "matakuliah",
                    "SEMESTER": "semester",
                }
                return str(dosen_info.get(key_map.get(placeholder, ""), "-"))

            # ===== INFO MATAKULIAH =====
            if intent in ["INFO_MATAKULIAH", "SKS_MATKUL", "PRASYARAT_MATKUL"]:
                matkul_name = self._normalize_alias_query(
                    self._normalize_entity_value(entities.get("MATA_KULIAH")), intent, placeholder
                )
                matkul_info = self.kb_service.get_matakuliah_detail(matkul_name)
                key_map = {
                    "MATA_KULIAH": "mata_kuliah",  # ✅ Perbaikan minor di sini
                    "KODE_MATAKULIAH": "kode",
                    "PRODI": "prodi",
                    "SEMESTER": "semester",
                    "SKS": "sks",
                    "PRASYARAT": "prasyarat",
                }
                return str(getattr(matkul_info, key_map.get(placeholder, ""), "-")) if matkul_info else "-"

            # ===== JADWAL SEMESTER =====
            if intent == "JADWAL_SEMESTER":
                semester = self._normalize_entity_value(entities.get("SEMESTER")) or "semester 1"
                kalender = self.kb_service.get_jadwal_semester(semester)
                key_map = {"TANGGAL_MULAI": "mulai", "TANGGAL_SELESAI": "selesai"}
                return getattr(kalender, key_map.get(placeholder, ""), "-") if kalender else "-"

            # ===== JADWAL =====
            if intent in ["JADWAL_MATAKULIAH", "JADWAL_HARI", "JADWAL_RUANGAN"]:
                matkul_name = self._normalize_alias_query(
                    self._normalize_entity_value(entities.get("MATA_KULIAH")), intent, placeholder
                )
                jadwal = self.kb_service.get_jadwal_by_matakuliah(matkul_name)
                key_map = {"HARI": "hari", "WAKTU": "jam", "RUANGAN": "ruang"}
                return jadwal.get(key_map.get(placeholder, ""), "-")

            # ===== JADWAL PRODI =====
            if intent in ["JADWAL_PRODI"]:
                prodi = self._normalize_entity_value(entities.get("PRODI")) or "Teknik Informatika"
                kalender = self.kb_service.get_kalender_akademik(prodi, self._normalize_entity_value(entities.get("SEMESTER")))
                key_map = {"TANGGAL_MULAI": "mulai", "TANGGAL_SELESAI": "selesai"}
                return getattr(kalender, key_map.get(placeholder, ""), "-") if kalender else "-"

            # ===== KONTAK & NIDN =====
            if intent in ["KONTAK_DOSEN", "NIDN_DOSEN"]:
                dosen_name = self._normalize_alias_query(
                    self._normalize_dosen_name(self._normalize_entity_value(entities.get("DOSEN"))), intent, placeholder
                )
                dosen_info = self.kb_service.get_dosen_info(dosen_name)
                return dosen_info.get("nidn", "-") if placeholder == "NIDN" else dosen_info.get("no_hp", "-")

            # ===== SYARAT SKRIPSI =====
            if intent == "SYARAT_SKRIPSI":
                prodi = self._normalize_entity_value(entities.get("PRODI")) or "Teknik Informatika"
                syarat = self.kb_service.get_syarat_skripsi(prodi)
                if syarat:
                    return str(syarat[0].ipk_minimum) if placeholder == "IPK" else str(syarat[0].sks_minimum)
            return "-"
        except Exception as e:
            logger.error(f"[LOOKUP ERROR] {e}")
            return "-"
