"""
Container security scanner.

Uses Trivy and custom checks for container security.
"""
import asyncio
import json
import hashlib
from typing import Optional
from datetime import datetime

from src.models.finding import Finding, Severity, FindingCategory
from src.access.docker_socket import DockerAccess, ContainerInfo


class ContainerScanner:
    """
    Scan containers and images for security issues.
    
    Capabilities:
    - Vulnerability scanning (via Trivy)
    - Secret detection in images/containers
    - Configuration security checks
    - Supply chain analysis
    """
    
    def __init__(self):
        self.docker = DockerAccess()
    
    async def scan_image(self, image_name: str) -> list[Finding]:
        """
        Scan a container image for vulnerabilities.
        
        Uses Trivy for CVE detection.
        """
        findings: list[Finding] = []
        
        # Run Trivy scan
        try:
            process = await asyncio.create_subprocess_exec(
                "trivy", "image",
                "--format", "json",
                "--severity", "CRITICAL,HIGH,MEDIUM",
                "--quiet",
                image_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300,
            )
            
            if stdout:
                result = json.loads(stdout.decode())
                findings.extend(self._parse_trivy_results(result, image_name))
        
        except asyncio.TimeoutError:
            pass
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error scanning image {image_name}: {e}")
        
        return findings
    
    async def scan_container(
        self,
        container_id: str,
        scan_secrets: bool = True,
        scan_config: bool = True,
    ) -> list[Finding]:
        """
        Scan a running container for security issues.
        
        Checks:
        - Environment variables for secrets
        - Mounted volumes for sensitive data
        - Running as root
        - Privileged mode
        """
        findings: list[Finding] = []
        
        # Get container details
        inspection = await self.docker.inspect_container(container_id)
        if not inspection:
            return findings
        
        container_name = inspection.get("Name", "").lstrip("/")
        config = inspection.get("Config", {})
        host_config = inspection.get("HostConfig", {})
        
        if scan_config:
            # Check if running as root
            user = config.get("User", "")
            if not user or user == "root" or user == "0":
                findings.append(Finding(
                    id=f"container-root-{container_id[:8]}",
                    title="Container Running as Root",
                    description=f"Container '{container_name}' is running as root user",
                    severity=Severity.MEDIUM,
                    category=FindingCategory.MISCONFIGURATION,
                    container_name=container_name,
                    scanner="container_scanner",
                    scan_time=datetime.utcnow(),
                    recommendation="Run container as non-root user. Add USER directive to Dockerfile.",
                ))
        
        # Check for privileged mode
        if scan_config and host_config.get("Privileged"):
            findings.append(Finding(
                id=f"container-privileged-{container_id[:8]}",
                title="Container Running in Privileged Mode",
                description=f"Container '{container_name}' is running with privileged flag",
                severity=Severity.CRITICAL,
                category=FindingCategory.MISCONFIGURATION,
                container_name=container_name,
                scanner="container_scanner",
                scan_time=datetime.utcnow(),
                recommendation="Remove --privileged flag. Use specific capabilities instead.",
            ))
        
        # Check for host network mode
        if scan_config and host_config.get("NetworkMode") == "host":
            findings.append(Finding(
                id=f"container-hostnet-{container_id[:8]}",
                title="Container Using Host Network",
                description=f"Container '{container_name}' is using host network mode",
                severity=Severity.HIGH,
                category=FindingCategory.MISCONFIGURATION,
                container_name=container_name,
                scanner="container_scanner",
                scan_time=datetime.utcnow(),
                recommendation="Use bridge network mode with explicit port mappings.",
            ))
        
        # Check for sensitive volume mounts
        if scan_config:
            mounts = inspection.get("Mounts", [])
            sensitive_paths = ["/etc", "/var/run/docker.sock", "/root", "/home"]
            
            for mount in mounts:
                source = mount.get("Source", "")
                for sensitive in sensitive_paths:
                    if source.startswith(sensitive):
                        rw = "read-write" if mount.get("RW", True) else "read-only"
                        severity = Severity.CRITICAL if "docker.sock" in source else Severity.MEDIUM
                        
                        findings.append(Finding(
                            id=f"container-mount-{hashlib.md5(source.encode()).hexdigest()[:8]}",
                            title=f"Sensitive Path Mounted: {source}",
                            description=f"Container '{container_name}' has {sensitive} mounted ({rw})",
                            severity=severity,
                            category=FindingCategory.MISCONFIGURATION,
                            container_name=container_name,
                            evidence=f"Mount: {source} -> {mount.get('Destination')} ({rw})",
                            scanner="container_scanner",
                            scan_time=datetime.utcnow(),
                            recommendation="Avoid mounting sensitive host paths. Use read-only where possible.",
                        ))
        
        # Check for dangerous capabilities
        if scan_config:
            cap_add = host_config.get("CapAdd", []) or []
            dangerous_caps = ["SYS_ADMIN", "NET_ADMIN", "SYS_PTRACE", "ALL"]
            
            for cap in cap_add:
                if cap in dangerous_caps:
                    findings.append(Finding(
                        id=f"container-cap-{cap.lower()}-{container_id[:8]}",
                        title=f"Dangerous Capability Added: {cap}",
                        description=f"Container '{container_name}' has {cap} capability",
                        severity=Severity.HIGH,
                        category=FindingCategory.MISCONFIGURATION,
                        container_name=container_name,
                        scanner="container_scanner",
                        scan_time=datetime.utcnow(),
                        recommendation=f"Remove {cap} capability unless absolutely required.",
                    ))
        
        # Check for missing healthcheck
        if scan_config:
            healthcheck = config.get("Healthcheck")
            if not healthcheck or healthcheck.get("Test", ["NONE"])[0] == "NONE":
                findings.append(Finding(
                    id=f"container-no-healthcheck-{container_id[:8]}",
                    title="No Healthcheck Configured",
                    description=f"Container '{container_name}' does not have a healthcheck defined",
                    severity=Severity.LOW,
                    category=FindingCategory.MISCONFIGURATION,
                    container_name=container_name,
                    scanner="container_scanner",
                    scan_time=datetime.utcnow(),
                    recommendation="Add HEALTHCHECK instruction to Dockerfile or use --health-cmd in docker run.",
                ))
        
        # Check for missing resource limits
        if scan_config:
            memory_limit = host_config.get("Memory", 0)
            cpu_quota = host_config.get("CpuQuota", 0)
            pids_limit = host_config.get("PidsLimit", 0)
            
            if memory_limit == 0:
                findings.append(Finding(
                    id=f"container-no-memlimit-{container_id[:8]}",
                    title="No Memory Limit Set",
                    description=f"Container '{container_name}' has no memory limit configured",
                    severity=Severity.MEDIUM,
                    category=FindingCategory.MISCONFIGURATION,
                    container_name=container_name,
                    scanner="container_scanner",
                    scan_time=datetime.utcnow(),
                    recommendation="Set memory limits using --memory flag to prevent resource exhaustion.",
                ))
            
            if cpu_quota == 0:
                findings.append(Finding(
                    id=f"container-no-cpulimit-{container_id[:8]}",
                    title="No CPU Limit Set",
                    description=f"Container '{container_name}' has no CPU limit configured",
                    severity=Severity.LOW,
                    category=FindingCategory.MISCONFIGURATION,
                    container_name=container_name,
                    scanner="container_scanner",
                    scan_time=datetime.utcnow(),
                    recommendation="Set CPU limits using --cpus or --cpu-quota flags.",
                ))
            
            if pids_limit == 0 or pids_limit == -1:
                findings.append(Finding(
                    id=f"container-no-pidlimit-{container_id[:8]}",
                    title="No PID Limit Set",
                    description=f"Container '{container_name}' has no PID limit configured",
                    severity=Severity.LOW,
                    category=FindingCategory.MISCONFIGURATION,
                    container_name=container_name,
                    scanner="container_scanner",
                    scan_time=datetime.utcnow(),
                    recommendation="Set PID limits using --pids-limit flag to prevent fork bombs.",
                ))
        
        # Check for read-only root filesystem
        if scan_config and not host_config.get("ReadonlyRootfs", False):
            findings.append(Finding(
                id=f"container-rw-rootfs-{container_id[:8]}",
                title="Writable Root Filesystem",
                description=f"Container '{container_name}' has a writable root filesystem",
                severity=Severity.LOW,
                category=FindingCategory.HARDENING,
                container_name=container_name,
                scanner="container_scanner",
                scan_time=datetime.utcnow(),
                recommendation="Use --read-only flag to make root filesystem read-only.",
            ))
        
        # Check for no-new-privileges
        if scan_config:
            security_opt = host_config.get("SecurityOpt", []) or []
            has_no_new_privs = any("no-new-privileges" in opt for opt in security_opt)
            if not has_no_new_privs:
                findings.append(Finding(
                    id=f"container-new-privs-{container_id[:8]}",
                    title="No-New-Privileges Not Set",
                    description=f"Container '{container_name}' can acquire new privileges",
                    severity=Severity.MEDIUM,
                    category=FindingCategory.HARDENING,
                    container_name=container_name,
                    scanner="container_scanner",
                    scan_time=datetime.utcnow(),
                    recommendation="Add --security-opt=no-new-privileges:true to prevent privilege escalation.",
                ))
        
        # Check for host PID namespace
        if scan_config and host_config.get("PidMode") == "host":
            findings.append(Finding(
                id=f"container-hostpid-{container_id[:8]}",
                title="Container Using Host PID Namespace",
                description=f"Container '{container_name}' shares the host PID namespace",
                severity=Severity.HIGH,
                category=FindingCategory.MISCONFIGURATION,
                container_name=container_name,
                scanner="container_scanner",
                scan_time=datetime.utcnow(),
                recommendation="Avoid using --pid=host unless absolutely required.",
            ))
        
        # Check for host IPC namespace
        if scan_config and host_config.get("IpcMode") == "host":
            findings.append(Finding(
                id=f"container-hostipc-{container_id[:8]}",
                title="Container Using Host IPC Namespace",
                description=f"Container '{container_name}' shares the host IPC namespace",
                severity=Severity.MEDIUM,
                category=FindingCategory.MISCONFIGURATION,
                container_name=container_name,
                scanner="container_scanner",
                scan_time=datetime.utcnow(),
                recommendation="Avoid using --ipc=host unless absolutely required.",
            ))
        
        # Check logging configuration
        if scan_config:
            log_config = host_config.get("LogConfig", {})
            log_type = log_config.get("Type", "")
            if log_type == "none":
                findings.append(Finding(
                    id=f"container-no-logging-{container_id[:8]}",
                    title="Container Logging Disabled",
                    description=f"Container '{container_name}' has logging disabled",
                    severity=Severity.HIGH,
                    category=FindingCategory.MISCONFIGURATION,
                    container_name=container_name,
                    scanner="container_scanner",
                    scan_time=datetime.utcnow(),
                    recommendation="Enable container logging for audit and troubleshooting.",
                ))
        
        # Scan for secrets in environment and configs
        if scan_secrets:
            secret_findings = await self.docker.scan_container_secrets(container_id)
            findings.extend(secret_findings)
        
        return findings
    
    async def scan_all_containers(self) -> list[dict]:
        """Scan all running containers."""
        results = []
        
        containers = await self.docker.list_containers(all=False)
        
        for container in containers:
            findings = await self.scan_container(container.id)
            results.append({
                "container": container.name,
                "image": container.image,
                "findings_count": len(findings),
                "findings": findings,
                "critical": sum(1 for f in findings if f.severity == Severity.CRITICAL),
                "high": sum(1 for f in findings if f.severity == Severity.HIGH),
            })
        
        return results
    
    def _parse_trivy_results(self, result: dict, image_name: str) -> list[Finding]:
        """Parse Trivy JSON output into findings."""
        findings = []
        
        for target in result.get("Results", []):
            target_name = target.get("Target", "")
            
            for vuln in target.get("Vulnerabilities", []):
                severity_map = {
                    "CRITICAL": Severity.CRITICAL,
                    "HIGH": Severity.HIGH,
                    "MEDIUM": Severity.MEDIUM,
                    "LOW": Severity.LOW,
                }
                
                severity = severity_map.get(vuln.get("Severity", "").upper(), Severity.MEDIUM)
                
                findings.append(Finding(
                    id=f"vuln-{vuln.get('VulnerabilityID', 'unknown')}",
                    title=f"{vuln.get('VulnerabilityID', 'CVE')}: {vuln.get('PkgName', 'Unknown Package')}",
                    description=vuln.get("Description", "")[:500],
                    severity=severity,
                    category=FindingCategory.VULNERABILITY,
                    cve_id=vuln.get("VulnerabilityID"),
                    cvss_score=vuln.get("CVSS", {}).get("nvd", {}).get("V3Score"),
                    file_path=f"{image_name}:{target_name}",
                    evidence=f"Package: {vuln.get('PkgName')} {vuln.get('InstalledVersion')} (fixed in {vuln.get('FixedVersion', 'N/A')})",
                    scanner="trivy",
                    scan_time=datetime.utcnow(),
                    recommendation=f"Upgrade {vuln.get('PkgName')} to {vuln.get('FixedVersion', 'latest version')}",
                    references=vuln.get("References", [])[:5],
                ))
        
        return findings
