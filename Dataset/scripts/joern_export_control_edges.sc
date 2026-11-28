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

val writer = Files.newBufferedWriter(Paths.get(outFile), StandardCharsets.UTF_8)
var first = true

def writeEdge(src: AstNode, dst: AstNode, edgeType: String): Unit = {
  if (src.lineNumber.nonEmpty &&
      dst.lineNumber.nonEmpty &&
      nodeFile(src).nonEmpty &&
      nodeFile(src) == nodeFile(dst)) {
    if (!first) writer.write(",")
    writer.write(s"""{"type":"$edgeType","src":${nodeJson(src)},"dst":${nodeJson(dst)}}""")
    first = false
  }
}

writer.write("[")

val roots =
  (cpg.call.l ++ cpg.controlStructure.l)
    .collect { case node: AstNode => node }
    .distinctBy(_.id)

roots.foreach { src =>
  src.controls.foreach { dst =>
    writeEdge(src, dst, "CONTROL_DEPENDENCE")
  }
}

val cfgNodes = cpg.all.collect { case node: CfgNode => node }.l

cfgNodes.foreach { src =>
  src.cfgNext.foreach { dst =>
    writeEdge(src, dst, "CFG_NEXT")
  }
}

writer.write("]")
writer.close()
