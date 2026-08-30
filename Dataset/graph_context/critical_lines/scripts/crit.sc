import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes._
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

importCpg(cpgFile)

def esc(value: String): String =
  value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

def nodeFile(node: AstNode): String = node.file.name.headOption.getOrElse("")

def nodeJson(node: AstNode): String = {
  val line = node.lineNumber.map(_.toString).getOrElse("null")
  s"""{"id":${node.id},"file":"${esc(nodeFile(node))}","line":$line,"code":"${esc(node.code)}"}"""
}

// CFG
val cfgEdges = cpg.cfgNode.l.collect { case n: AstNode => n }.flatMap { src =>
  src.cfgNext.l.collect {
    case dst: AstNode if src.lineNumber.nonEmpty && dst.lineNumber.nonEmpty &&
      nodeFile(src).nonEmpty && nodeFile(src) == nodeFile(dst) =>
      s"""{"type":"CFG_NEXT","src":${nodeJson(src)},"dst":${nodeJson(dst)}}"""
  }
}.distinct

// AST
val astEdges = cpg.astNode.l.collect { case n: AstNode => n }.flatMap { src =>
  src.astChildren.l.collect {
    case dst: AstNode if src.lineNumber.nonEmpty && dst.lineNumber.nonEmpty &&
      nodeFile(src).nonEmpty && nodeFile(src) == nodeFile(dst) =>
      s"""{"type":"AST_CHILD","src":${nodeJson(src)},"dst":${nodeJson(dst)}}"""
  }
}.distinct

val edges = (cfgEdges ++ astEdges).distinct
Files.write(Paths.get(outFile), ("[" + edges.mkString(",") + "]").getBytes(StandardCharsets.UTF_8))
println(s"Exported ${edges.size} edges (CFG=${cfgEdges.size} AST=${astEdges.size})")