"""
Compliance Mapper

Maps security findings to compliance framework controls.
Uses both pattern matching and AI-enhanced semantic mapping.
"""

import re
from dataclasses import dataclass
from typing import Optional
import httpx

from .frameworks import (
    ComplianceFramework,
    Control,
    ControlStatus,
    ControlPriority,
    FrameworkType,
    get_framework,
)
from src.models.finding import Finding, Severity, FindingCategory
from src.config import Settings


@dataclass
class MappingResult:
    """Result of mapping a finding to controls"""
    finding: Finding
    matched_controls: list[tuple[Control, float]]  # (control, confidence)
    primary_control: Optional[Control] = None
    confidence: float = 0.0


class ComplianceMapper:
    """
    Maps security findings to compliance framework controls.
    
    Uses multiple strategies:
    1. Category matching - finding categories -> control categories
    2. Pattern matching - regex patterns in finding content
    3. Severity alignment - critical findings -> critical controls
    4. AI semantic matching (optional) - embeddings-based similarity
    """
    
    def __init__(
        self, 
        framework: ComplianceFramework,
        use_ai: bool = True,
        valet_url: str = None
    ):
        self.framework = framework
        self.use_ai = use_ai
        self.settings = Settings()
        self.valet_url = valet_url or self.settings.valet_runtime_url
        self.rerank_url = self.settings.valet_rerank_url or self.settings.valet_runtime_url
        self.rerank_model = self.settings.valet_rerank_model
        self.enable_reranking = self.settings.enable_reranking
        
        # Build category index for fast lookup
        self._category_index: dict[str, list[Control]] = {}
        self._pattern_index: dict[str, list[Control]] = {}
        self._build_indexes()
    
    def _build_indexes(self):
        """Build indexes for fast control lookup"""
        for control in self.framework.controls:
            # Index by finding categories
            for cat in control.finding_categories:
                if cat not in self._category_index:
                    self._category_index[cat] = []
                self._category_index[cat].append(control)
            
            # Index by patterns
            for pattern in control.finding_patterns:
                if pattern not in self._pattern_index:
                    self._pattern_index[pattern] = []
                self._pattern_index[pattern].append(control)
    
    def map_finding(self, finding: Finding) -> MappingResult:
        """Map a single finding to relevant compliance controls"""
        matched: dict[str, tuple[Control, float]] = {}
        
        # Strategy 1: Category matching (highest weight)
        category_matches = self._match_by_category(finding)
        for control, conf in category_matches:
            key = control.id
            if key not in matched or matched[key][1] < conf:
                matched[key] = (control, conf)
        
        # Strategy 2: Pattern matching in title/description
        pattern_matches = self._match_by_pattern(finding)
        for control, conf in pattern_matches:
            key = control.id
            if key not in matched:
                matched[key] = (control, conf)
            else:
                # Boost confidence if matched by multiple strategies
                existing_conf = matched[key][1]
                matched[key] = (control, min(1.0, existing_conf + conf * 0.3))
        
        # Strategy 3: Severity alignment boost
        for key, (control, conf) in list(matched.items()):
            severity_boost = self._severity_alignment(finding.severity, control.priority)
            matched[key] = (control, min(1.0, conf + severity_boost))
        
        # Sort by confidence
        sorted_matches = sorted(matched.values(), key=lambda x: x[1], reverse=True)
        
        return MappingResult(
            finding=finding,
            matched_controls=sorted_matches,
            primary_control=sorted_matches[0][0] if sorted_matches else None,
            confidence=sorted_matches[0][1] if sorted_matches else 0.0
        )
    
    def _match_by_category(self, finding: Finding) -> list[tuple[Control, float]]:
        """Match finding by its category"""
        results = []
        
        # Get category as string
        category = finding.category.value if hasattr(finding.category, 'value') else str(finding.category)
        
        # Direct category match
        if category in self._category_index:
            for control in self._category_index[category]:
                results.append((control, 0.8))  # High confidence for category match
        
        # Partial category matching (e.g., "secrets" matches "secrets_detection")
        for cat_key, controls in self._category_index.items():
            if cat_key in category or category in cat_key:
                for control in controls:
                    if (control, 0.8) not in results:
                        results.append((control, 0.6))
        
        return results
    
    def _match_by_pattern(self, finding: Finding) -> list[tuple[Control, float]]:
        """Match finding by pattern in title/description"""
        results = []
        
        # Combine searchable text
        location = finding.file_path or finding.container_name or finding.endpoint or ""
        search_text = f"{finding.title} {finding.description} {location}".lower()
        
        for pattern, controls in self._pattern_index.items():
            # Case-insensitive pattern matching
            if pattern.lower() in search_text:
                for control in controls:
                    results.append((control, 0.6))
            # Regex pattern matching for more complex patterns
            elif re.search(re.escape(pattern), search_text, re.IGNORECASE):
                for control in controls:
                    results.append((control, 0.5))
        
        return results
    
    def _severity_alignment(self, finding_severity: Severity, control_priority: ControlPriority) -> float:
        """Calculate severity alignment boost"""
        severity_map = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
            Severity.INFO: 0,
        }
        
        priority_map = {
            ControlPriority.CRITICAL: 4,
            ControlPriority.HIGH: 3,
            ControlPriority.MEDIUM: 2,
            ControlPriority.LOW: 1,
        }
        
        sev_score = severity_map.get(finding_severity, 2)
        pri_score = priority_map.get(control_priority, 2)
        
        # Boost when severities align
        diff = abs(sev_score - pri_score)
        if diff == 0:
            return 0.15
        elif diff == 1:
            return 0.05
        else:
            return 0.0
    
    def map_findings(self, findings: list[Finding]) -> list[MappingResult]:
        """Map multiple findings to controls"""
        return [self.map_finding(f) for f in findings]
    
    async def map_finding_with_ai(self, finding: Finding) -> MappingResult:
        """
        Enhanced mapping using AI embeddings for semantic similarity.
        Falls back to rule-based mapping if AI is unavailable.
        """
        # Start with rule-based mapping
        result = self.map_finding(finding)
        
        if not self.use_ai:
            return result
        
        try:
            # Get embedding for finding
            finding_text = f"{finding.title}: {finding.description}"
            finding_embedding = await self._get_embedding(finding_text)
            
            if not finding_embedding:
                return result
            
            # Get embeddings for unmatched controls and find semantic matches
            unmatched_controls = [
                c for c in self.framework.controls 
                if c.id not in {m[0].id for m in result.matched_controls}
            ]
            
            for control in unmatched_controls:
                control_text = f"{control.name}: {control.description}"
                control_embedding = await self._get_embedding(control_text)
                
                if control_embedding:
                    similarity = self._cosine_similarity(finding_embedding, control_embedding)
                    
                    # If semantically similar, add to matches
                    if similarity > 0.7:
                        result.matched_controls.append((control, similarity * 0.8))
            
            # Re-sort by confidence
            result.matched_controls.sort(key=lambda x: x[1], reverse=True)
            
            if result.matched_controls:
                result.primary_control = result.matched_controls[0][0]
                result.confidence = result.matched_controls[0][1]
            
            if self.enable_reranking and result.matched_controls:
                reranked = await self._rerank_controls(finding, result.matched_controls)
                if reranked:
                    result.matched_controls = reranked
                    result.primary_control = result.matched_controls[0][0]
                    result.confidence = result.matched_controls[0][1]
        
        except Exception as e:
            # Log error but continue with rule-based results
            print(f"AI mapping failed: {e}")
        
        return result

    async def _rerank_controls(
        self,
        finding: Finding,
        matches: list[tuple[Control, float]],
    ) -> Optional[list[tuple[Control, float]]]:
        """Rerank matched controls using Valet reranker."""
        if not matches:
            return None
        
        location = finding.file_path or finding.container_name or finding.endpoint or ""
        query = f"{finding.title}\n{finding.description}\n{location}".strip()
        documents = [
            f"{control.name}\n{control.description}\nCategory: {control.category}"
            for control, _ in matches
        ]
        
        try:
            payload = {"query": query, "documents": documents}
            if self.rerank_model:
                payload["model"] = self.rerank_model
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.rerank_url}/v1/rerank",
                    json=payload,
                )
            
            if response.status_code != 200:
                return None
            
            ranked = self._parse_rerank_response(response.json())
            if not ranked:
                return None
            
            index_to_score = {r["index"]: r.get("score") for r in ranked}
            ordered = []
            for r in ranked:
                idx = r["index"]
                if idx < len(matches):
                    control, confidence = matches[idx]
                    rerank_score = index_to_score.get(idx)
                    if rerank_score is not None:
                        confidence = max(confidence, float(rerank_score))
                    ordered.append((control, confidence))
            return ordered
        except Exception:
            return None

    def _parse_rerank_response(self, data: dict) -> list[dict]:
        """Normalize rerank response into list of {index, score}."""
        if not data:
            return []
        
        results = None
        if isinstance(data, dict):
            for key in ("results", "data", "rerank"):
                if key in data and isinstance(data[key], list):
                    results = data[key]
                    break
        
        if not results:
            return []
        
        normalized = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index") if "index" in item else item.get("document_index")
            score = item.get("relevance_score", item.get("score"))
            if index is None:
                continue
            normalized.append({"index": int(index), "score": score})
        
        normalized.sort(key=lambda x: x.get("score", 0), reverse=True)
        return normalized
    
    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding from Valet"""
        try:
            payload = {
                "input": text,
                "model": self.settings.valet_embedding_model,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.valet_url}/v1/embeddings",
                    json=payload,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["data"][0]["embedding"]
        except Exception:
            pass
        
        return None
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


def create_mapper(framework_type: FrameworkType, use_ai: bool = True) -> ComplianceMapper:
    """Factory function to create a compliance mapper for a specific framework"""
    framework = get_framework(framework_type)
    return ComplianceMapper(framework, use_ai=use_ai)


# Convenience mappers for common frameworks
def create_nist_csf_mapper(use_ai: bool = True) -> ComplianceMapper:
    return create_mapper(FrameworkType.NIST_CSF, use_ai)


def create_hipaa_mapper(use_ai: bool = True) -> ComplianceMapper:
    return create_mapper(FrameworkType.HIPAA, use_ai)


def create_essential8_mapper(use_ai: bool = True) -> ComplianceMapper:
    return create_mapper(FrameworkType.ESSENTIAL_8, use_ai)


def create_pci_dss_mapper(use_ai: bool = True) -> ComplianceMapper:
    return create_mapper(FrameworkType.PCI_DSS, use_ai)


def create_cis_docker_mapper(use_ai: bool = True) -> ComplianceMapper:
    return create_mapper(FrameworkType.CIS_DOCKER, use_ai)
