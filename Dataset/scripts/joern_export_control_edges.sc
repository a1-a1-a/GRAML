import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes._
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

importCpg(cpgFile)

def esc(value: String): String =
  value
    .replace("\\", "\\\\")
    .replace("\"", "\\\"")
    .replace("\n", "\\n")
    .replace("\r", "\\r")
    .replace("\t", "\\t")

def nodeFile(node: AstNode): String =
  node.file.name.headOption.getOrElse("")

def nodeJson(node: AstNode): String = {
  val line = node.lineNumber.map(_.toString).getOrElse("null")
  s"""{"id":${node.id},"file":"${esc(nodeFile(node))}","line":$line,"code":"${esc(node.code)}"}"""
}

val roots =
  (cpg.call.l ++ cpg.controlStructure.l)
    .collect { case node: AstNode => node }
    .distinctBy(_.id)

val controlEdges = roots.flatMap { src =>
  src.controls.l.collect {
    case dst: AstNode
      if src.lineNumber.nonEmpty &&
         dst.lineNumber.nonEmpty &&
         nodeFile(src).nonEmpty &&
         nodeFile(src) == nodeFile(dst) =>
      s"""{"type":"CONTROL_DEPENDENCE","src":${nodeJson(src)},"dst":${nodeJson(dst)}}"""
  }
}

val cfgEdges = cpg.cfgNode.l
  .collect { case node: AstNode => node }
  .flatMap { src =>
    src.cfgNext.l.collect {
      case dst: AstNode
        if src.lineNumber.nonEmpty &&
           dst.lineNumber.nonEmpty &&
           nodeFile(src).nonEmpty &&
           nodeFile(src) == nodeFile(dst) =>
        s"""{"type":"CFG_NEXT","src":${nodeJson(src)},"dst":${nodeJson(dst)}}"""
    }
  }

val astEdges = cpg.astNode.l
  .collect { case node: AstNode => node }
  .flatMap { src =>
    src.astChildren.l.collect {
      case dst: AstNode
        if src.lineNumber.nonEmpty &&
           dst.lineNumber.nonEmpty &&
           nodeFile(src).nonEmpty &&
           nodeFile(src) == nodeFile(dst) =>
        s"""{"type":"AST_CHILD","src":${nodeJson(src)},"dst":${nodeJson(dst)}}"""
    }
  }

val edges = (controlEdges ++ cfgEdges ++ astEdges).distinct

Files.write(
  Paths.get(outFile),
  ("[" + edges.mkString(",") + "]").getBytes(StandardCharsets.UTF_8)
)
