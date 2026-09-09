"""El reporte de ingesta. **El silencio no es éxito.**

Si el pipeline dejó algo afuera —un documento abortado, una página vacía, un texto marcado
como sospechoso— tiene que estar escrito. Un resumen que sólo dice «13 documentos OK» se lee
como cobertura total, y ese es el modo de fallo caro: nueve bugs devolviendo «listo» con
datos malos.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Reporte:
    procesados: list[str] = field(default_factory=list)
    abortados: list[tuple[str, str]] = field(default_factory=list)
    marcados: list[tuple[str, str]] = field(default_factory=list)
    omitidos: list[tuple[str, str]] = field(default_factory=list)
    chunks: int = 0
    regenerados: int = 0
    sin_ground_truth: list[str] = field(default_factory=list)
    caracteres: int = 0
    versiones: dict[str, str | int] = field(default_factory=dict)
    canario: dict[str, float | int] | None = None

    def a_markdown(self) -> str:
        lineas = [
            "# Reporte de ingesta",
            "",
            "| | |",
            "|---|---|",
            f"| Procesados | {len(self.procesados)} |",
            f"| Abortados | {len(self.abortados)} |",
            f"| Marcados | {len(self.marcados)} |",
            f"| Omitidos por catálogo | {len(self.omitidos)} |",
            f"| Chunks | {self.chunks} |",
            f"| Oro reemplazado | {self.regenerados} documento(s) de la corrida anterior |",
            f"| Caracteres | {self.caracteres} |",
            "",
            "## Versiones",
            "",
        ]
        for clave, valor in sorted(self.versiones.items()):
            lineas.append(f"- `{clave}`: `{valor}`")
        lineas.append("")
        lineas.append(
            "> Una métrica sin estas tres versiones al lado no es comparable con la anterior."
        )

        for titulo, filas, nota in (
            ("Abortados", self.abortados, "No produjeron ningún chunk."),
            ("Marcados", self.marcados, "Se ingirieron, pero con una señal de sospecha."),
            ("Omitidos por catálogo", self.omitidos, "Excluidos o sin acceso abierto."),
        ):
            lineas += ["", f"## {titulo}", ""]
            if not filas:
                lineas.append("_Ninguno._")
                continue
            lineas.append(f"_{nota}_")
            lineas += ["", "| id | motivo |", "|---|---|"]
            lineas += [f"| `{doc_id}` | {motivo} |" for doc_id, motivo in filas]

        if self.canario is not None:
            fr = self.canario.get("falso_rechazo")
            lineas += ["", "## Canario", ""]
            if not self.canario.get("midio"):
                lineas.append(
                    "🚨 **El canario no midió.** Un juez desconectado no es un juez "
                    "satisfecho: este resultado no vale como verde."
                )
            lineas += [
                f"- Oraciones evaluadas: **{self.canario['oraciones']}**",
                f"- Verifican: **{self.canario['verifican']}**",
                f"- Falso rechazo: **{fr:.1%}** (línea base 15,5%)" if fr is not None
                else "- Falso rechazo: **no medido**",
            ]
            sin_gt = self.canario.get("sin_ground_truth") or []
            if sin_gt:
                lineas.append(
                    f"- ⚠️ Sin ground truth, **no auditados** ({len(sin_gt)}): "
                    + ", ".join(f"`{d}`" for d in sin_gt)
                )
            lineas += [
                "",
                "> Ground truth externo: abstracts oficiales de arXiv y Crossref. Detecta que "
                "el crudo dejó de ser fiel al PDF, cosa que las invariantes no ven.",
            ]

        lineas += ["", "## Procesados", ""]
        lineas += [f"- `{d}`" for d in self.procesados] or ["_Ninguno._"]
        return "\n".join(lineas) + "\n"
