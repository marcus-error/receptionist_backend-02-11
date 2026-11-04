"""
Asterisk Dynamic Configuration Generator for Multi-Tenant AI Voice Receptionist.

Automatically generates PJSIP endpoints, auth digest credentials, AORs,
and dialplan routes dynamically whenever frontend users submit SIP trunk configuration.
Preserves multi-tenant isolation across all configured organizations.
"""
import os
import sys
import logging
import subprocess
from typing import Optional, Dict, Any, List

logger = logging.getLogger("ai-receptionist.asterisk_configurator")


class AsteriskConfigurator:
    """Manages dynamic Asterisk PJSIP configuration and dialplan updates across all tenants."""

    @staticmethod
    def _is_wsl_or_linux() -> bool:
        """Check if Linux or WSL is available."""
        if sys.platform != "win32":
            return True
        try:
            res = subprocess.run(["wsl.exe", "--", "which", "asterisk"], capture_output=True, text=True, timeout=2)
            return res.returncode == 0
        except Exception:
            return False

    @classmethod
    def build_multi_tenant_config(
        cls,
        current_org_id: int,
        sip_host: str,
        sip_username: str,
        sip_password: str,
        sip_port: int = 5060,
        did_number: Optional[str] = None,
        all_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """
        Build complete multi-tenant pjsip.conf and extensions.conf contents.
        Combines all active organization configurations so saving one tenant does not
        overwrite another tenant's endpoints or dialplans.
        """
        # Map of org_id -> dict of parameters
        orgs_map: Dict[int, Dict[str, Any]] = {}

        if all_configs:
            for cfg in all_configs:
                oid = cfg.get("organization_id")
                if oid:
                    orgs_map[int(oid)] = cfg

        # Current org parameters take precedence
        orgs_map[int(current_org_id)] = {
            "organization_id": int(current_org_id),
            "sip_host": (sip_host or "127.0.0.1").strip(),
            "sip_username": (sip_username or f"agent_{current_org_id}").strip(),
            "sip_password": sip_password or "",
            "sip_port": sip_port or 5060,
            "did_number": (did_number or "").strip(),
        }

        # Build pjsip.conf
        pjsip_lines = [
            "; =====================================================================",
            "; Dynamically Generated Multi-Tenant PJSIP Configuration",
            "; =====================================================================",
            "",
            "[global]",
            "type=global",
            "user_agent=Asterisk-AI-Receptionist",
            "",
            "[transport-udp]",
            "type=transport",
            "protocol=udp",
            "bind=0.0.0.0:5060",
            "",
        ]

        # Build extensions.conf
        dialplan_lines = [
            "; =====================================================================",
            "; Dynamically Generated Multi-Tenant Dialplan Configuration",
            "; =====================================================================",
            "",
            "[general]",
            "static=yes",
            "writeprotect=no",
            "",
        ]

        for oid in sorted(orgs_map.keys()):
            odata = orgs_map[oid]
            u_name = odata.get("sip_username") or f"agent_{oid}"
            u_pass = odata.get("sip_password") or ""
            did = odata.get("did_number") or ""
            ep_name = f"org_{oid}_ep"
            auth_name = f"auth_org_{oid}"
            aor_name = f"aor_org_{oid}"
            ctx_name = f"context_org_{oid}"

            pjsip_lines.extend([
                f"; --- Organization #{oid} ---",
                f"[{ep_name}]",
                "type=endpoint",
                f"context={ctx_name}",
                "disallow=all",
                "allow=ulaw,alaw,opus,gsm",
                f"auth={auth_name}",
                f"aors={aor_name}",
                "direct_media=no",
                "",
                f"[{auth_name}]",
                "type=auth",
                "auth_type=userpass",
                f"username={u_name}",
                f"password={u_pass}",
                "",
                f"[{aor_name}]",
                "type=aor",
                "max_contacts=10",
                "remove_existing=yes",
                "",
            ])

            did_pattern = did if did else "_X."
            dialplan_lines.extend([
                f"; --- Organization #{oid} Dialplan ---",
                f"[{ctx_name}]",
                f"exten => {u_name},1,NoOp(=== Inbound Call to Org #{oid} AI Receptionist ===)",
                " same => n,Answer()",
                f' same => n,System(curl -s -X POST http://127.0.0.1:8000/api/v1/telephony/sip/inbound -H "Content-Type: application/json" -d \'{{"call_sid":"sip_${{UNIQUEID}}","from_number":"${{CALLERID(num)}}","org_id":{oid}}}\')',
                " same => n,Hangup()",
                "",
                f"exten => {did_pattern},1,NoOp(=== Inbound DID Call to Org #{oid} ===)",
                " same => n,Answer()",
                f' same => n,System(curl -s -X POST http://127.0.0.1:8000/api/v1/telephony/sip/inbound -H "Content-Type: application/json" -d \'{{"call_sid":"sip_${{UNIQUEID}}","from_number":"${{CALLERID(num)}}","org_id":{oid}}}\')',
                " same => n,Hangup()",
                "",
            ])

        return {
            "pjsip_content": "\n".join(pjsip_lines) + "\n",
            "dialplan_content": "\n".join(dialplan_lines) + "\n",
        }

    @classmethod
    def sync_sip_config(
        cls,
        org_id: int,
        user_id: int,
        sip_host: str,
        sip_username: str,
        sip_password: str,
        sip_port: int = 5060,
        did_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dynamically generate PJSIP endpoint and dialplan configuration for Asterisk.
        Enforces multi-tenant isolation by writing all tenant endpoints to pjsip.conf
        without overwriting existing organizations.
        """
        endpoint_name = f"org_{org_id}_ep"
        context_name = f"context_org_{org_id}"

        # Fetch other configured organizations from database
        all_configs: List[Dict[str, Any]] = []
        try:
            from src.config.database import SessionLocal
            from src.models.sip_config import SipConfig
            from src.utils.security import decrypt_secret

            db = SessionLocal()
            try:
                cfgs = db.query(SipConfig).all()
                for c in cfgs:
                    if not c.organization_id:
                        continue
                    plain_pass = decrypt_secret(c.sip_password_encrypted) if c.sip_password_encrypted else ""
                    all_configs.append({
                        "organization_id": c.organization_id,
                        "sip_host": c.sip_host or c.domain or c.sip_server or "127.0.0.1",
                        "sip_username": c.sip_username or c.login or f"agent_{c.organization_id}",
                        "sip_password": plain_pass,
                        "sip_port": c.sip_port or 5060,
                        "did_number": c.did_number or "",
                    })
            finally:
                db.close()
        except Exception as db_err:
            logger.warning(f"[AsteriskConfigurator] Could not query other org configs: {db_err}")

        config_docs = cls.build_multi_tenant_config(
            current_org_id=org_id,
            sip_host=sip_host,
            sip_username=sip_username,
            sip_password=sip_password,
            sip_port=sip_port,
            did_number=did_number,
            all_configs=all_configs,
        )

        pjsip_content = config_docs["pjsip_content"]
        dialplan_content = config_docs["dialplan_content"]

        # Write to Linux/WSL Asterisk configuration
        try:
            if sys.platform == "win32":
                import base64
                pjsip_b64 = base64.b64encode(pjsip_content.encode('utf-8')).decode('ascii')
                dialplan_b64 = base64.b64encode(dialplan_content.encode('utf-8')).decode('ascii')

                cmd_pjsip = f"echo '{pjsip_b64}' | base64 -d > /etc/asterisk/pjsip.conf"
                cmd_dialplan = f"echo '{dialplan_b64}' | base64 -d > /etc/asterisk/extensions.conf"

                subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-c", cmd_pjsip], check=True, timeout=15)
                subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-c", cmd_dialplan], check=True, timeout=15)

                subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "asterisk", "-rx", "module reload res_pjsip.so"], check=True, timeout=15)
                subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "asterisk", "-rx", "dialplan reload"], check=True, timeout=15)
            else:
                os.makedirs("/etc/asterisk/pjsip.d", exist_ok=True)
                with open("/etc/asterisk/pjsip.conf", "w") as f:
                    f.write(pjsip_content)
                with open("/etc/asterisk/extensions.conf", "w") as f:
                    f.write(dialplan_content)
                subprocess.run(["asterisk", "-rx", "module reload res_pjsip.so"], timeout=5)
                subprocess.run(["asterisk", "-rx", "dialplan reload"], timeout=5)

            logger.info(f"[AsteriskConfigurator] Dynamically synchronized multi-tenant PJSIP config. Org #{org_id} context '{context_name}'.")
            return {
                "success": True,
                "endpoint": endpoint_name,
                "context": context_name,
                "message": f"Asterisk multi-tenant PJSIP configuration synchronized for Org #{org_id}."
            }
        except Exception as err:
            logger.warning(f"[AsteriskConfigurator] Asterisk config sync error: {err}")
            return {
                "success": False,
                "error": str(err),
                "message": f"Could not sync Asterisk files: {err}"
            }
