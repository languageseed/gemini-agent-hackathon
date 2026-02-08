"""
Network security scanner.

Port scanning, service detection, and vulnerability probing.
"""
import asyncio
import json
import hashlib
import xml.etree.ElementTree as ET
from typing import Optional
from datetime import datetime

from src.models.finding import Finding, Severity, FindingCategory


class NetworkScanner:
    """
    Network security scanning.
    
    Capabilities:
    - Port scanning (nmap)
    - Service detection
    - Vulnerability detection (Nuclei)
    - SSL/TLS analysis
    """
    
    # Ports that are often risky when exposed
    RISKY_PORTS = {
        21: ("FTP", Severity.HIGH, "Unencrypted file transfer"),
        23: ("Telnet", Severity.CRITICAL, "Unencrypted remote access"),
        25: ("SMTP", Severity.MEDIUM, "Email server, may allow relay"),
        53: ("DNS", Severity.LOW, "DNS server"),
        110: ("POP3", Severity.HIGH, "Unencrypted email"),
        135: ("RPC", Severity.HIGH, "Windows RPC"),
        139: ("NetBIOS", Severity.HIGH, "Windows file sharing"),
        143: ("IMAP", Severity.HIGH, "Unencrypted email"),
        445: ("SMB", Severity.HIGH, "Windows file sharing"),
        512: ("rexec", Severity.CRITICAL, "Remote execution"),
        513: ("rlogin", Severity.CRITICAL, "Remote login"),
        514: ("rsh", Severity.CRITICAL, "Remote shell"),
        1433: ("MSSQL", Severity.HIGH, "Database server"),
        1521: ("Oracle", Severity.HIGH, "Database server"),
        3306: ("MySQL", Severity.HIGH, "Database server"),
        3389: ("RDP", Severity.HIGH, "Remote desktop"),
        5432: ("PostgreSQL", Severity.HIGH, "Database server"),
        5900: ("VNC", Severity.HIGH, "Remote desktop"),
        6379: ("Redis", Severity.HIGH, "Cache/database, often no auth"),
        11211: ("Memcached", Severity.HIGH, "Cache, often no auth"),
        27017: ("MongoDB", Severity.HIGH, "Database, may lack auth"),
    }
    
    async def scan_host(
        self,
        host: str,
        ports: str = "1-1000",
        scan_type: str = "quick",
    ) -> list[Finding]:
        """
        Scan a host for open ports and services.
        
        Args:
            host: Target hostname or IP
            ports: Port range (e.g., "1-1000", "22,80,443")
            scan_type: "quick" (top 1000) or "full" (specified range)
        """
        findings: list[Finding] = []
        
        # Run nmap scan
        nmap_findings = await self._run_nmap(host, ports, scan_type)
        findings.extend(nmap_findings)
        
        # Run Nuclei for vulnerability detection
        nuclei_findings = await self._run_nuclei(host)
        findings.extend(nuclei_findings)
        
        return findings
    
    async def _run_nmap(
        self,
        host: str,
        ports: str,
        scan_type: str,
    ) -> list[Finding]:
        """Run nmap port scan."""
        findings = []
        
        # Build nmap command
        if scan_type == "quick":
            cmd = ["nmap", "-sS", "-sV", "--top-ports", "1000", "-oX", "-", host]
        else:
            cmd = ["nmap", "-sS", "-sV", "-p", ports, "-oX", "-", host]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300,
            )
            
            if stdout:
                # Parse nmap XML output
                root = ET.fromstring(stdout.decode())
                
                for host_elem in root.findall(".//host"):
                    for port in host_elem.findall(".//port"):
                        port_id = int(port.get("portid", 0))
                        protocol = port.get("protocol", "tcp")
                        state = port.find("state")
                        service = port.find("service")
                        
                        if state is not None and state.get("state") == "open":
                            service_name = service.get("name", "unknown") if service is not None else "unknown"
                            service_version = service.get("version", "") if service is not None else ""
                            product = service.get("product", "") if service is not None else ""
                            
                            # Determine severity based on port
                            if port_id in self.RISKY_PORTS:
                                name, severity, reason = self.RISKY_PORTS[port_id]
                                findings.append(Finding(
                                    id=f"port-risky-{port_id}-{host}",
                                    title=f"Risky Port Open: {port_id}/{protocol} ({name})",
                                    description=f"{reason}. Service: {service_name} {product} {service_version}".strip(),
                                    severity=severity,
                                    category=FindingCategory.NETWORK,
                                    endpoint=f"{host}:{port_id}",
                                    evidence=f"Service: {service_name} {product} {service_version}",
                                    scanner="nmap",
                                    scan_time=datetime.utcnow(),
                                    recommendation=self._get_port_recommendation(port_id),
                                ))
                            else:
                                # Just informational for other open ports
                                findings.append(Finding(
                                    id=f"port-open-{port_id}-{host}",
                                    title=f"Open Port: {port_id}/{protocol} ({service_name})",
                                    description=f"Port {port_id} is open running {service_name} {service_version}",
                                    severity=Severity.INFO,
                                    category=FindingCategory.NETWORK,
                                    endpoint=f"{host}:{port_id}",
                                    evidence=f"Service: {service_name} {product} {service_version}",
                                    scanner="nmap",
                                    scan_time=datetime.utcnow(),
                                ))
        
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            print(f"Error running nmap: {e}")
        
        return findings
    
    async def _run_nuclei(self, host: str) -> list[Finding]:
        """Run Nuclei vulnerability scanner."""
        findings = []
        
        # Determine if HTTP/HTTPS
        targets = [f"http://{host}", f"https://{host}"]
        
        for target in targets:
            try:
                process = await asyncio.create_subprocess_exec(
                    "nuclei",
                    "-u", target,
                    "-t", "cves/",
                    "-t", "vulnerabilities/",
                    "-t", "misconfiguration/",
                    "-severity", "critical,high,medium",
                    "-json",
                    "-silent",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=600,
                )
                
                if stdout:
                    for line in stdout.decode().strip().split("\n"):
                        if line:
                            try:
                                vuln = json.loads(line)
                                
                                severity_map = {
                                    "critical": Severity.CRITICAL,
                                    "high": Severity.HIGH,
                                    "medium": Severity.MEDIUM,
                                    "low": Severity.LOW,
                                    "info": Severity.INFO,
                                }
                                
                                severity = severity_map.get(
                                    vuln.get("info", {}).get("severity", "medium").lower(),
                                    Severity.MEDIUM
                                )
                                
                                findings.append(Finding(
                                    id=f"nuclei-{vuln.get('template-id', 'unknown')}-{host}",
                                    title=vuln.get("info", {}).get("name", "Unknown Vulnerability"),
                                    description=vuln.get("info", {}).get("description", "")[:500],
                                    severity=severity,
                                    category=FindingCategory.VULNERABILITY,
                                    endpoint=vuln.get("matched-at", target),
                                    cve_id=vuln.get("info", {}).get("cve-id"),
                                    evidence=str(vuln.get("extracted-results", []))[:200],
                                    scanner="nuclei",
                                    scan_time=datetime.utcnow(),
                                    references=vuln.get("info", {}).get("reference", [])[:5],
                                ))
                            except json.JSONDecodeError:
                                pass
            
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"Error running nuclei: {e}")
        
        return findings
    
    def _get_port_recommendation(self, port: int) -> str:
        """Get remediation recommendation for port."""
        recommendations = {
            21: "Disable FTP. Use SFTP or SCP instead.",
            23: "Disable Telnet immediately. Use SSH instead.",
            25: "Ensure SMTP is not an open relay. Use authentication.",
            110: "Use POP3S (port 995) with TLS instead.",
            143: "Use IMAPS (port 993) with TLS instead.",
            135: "Block RPC from external access.",
            139: "Block NetBIOS from external access.",
            445: "Block SMB from external access or use SMB 3.0 with encryption.",
            512: "Disable rexec. Use SSH instead.",
            513: "Disable rlogin. Use SSH instead.",
            514: "Disable rsh. Use SSH instead.",
            1433: "Restrict SQL Server access to application servers only.",
            1521: "Restrict Oracle access to application servers only.",
            3306: "Restrict MySQL access. Enable SSL and require authentication.",
            3389: "Use VPN for RDP access. Enable NLA.",
            5432: "Restrict PostgreSQL access. Enable SSL.",
            5900: "Use VPN or SSH tunnel for VNC access.",
            6379: "Enable Redis AUTH. Bind to localhost or use TLS.",
            11211: "Bind Memcached to localhost. Use SASL authentication.",
            27017: "Enable MongoDB authentication. Bind to localhost.",
        }
        return recommendations.get(port, "Review if this port needs to be exposed.")
