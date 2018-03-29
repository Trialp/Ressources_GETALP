/* Generated By:JavaCC: Do not edit this line. UNLParserDebug.java */
package fr.imag.clips.papillon.business.pivax.unl_graph;

import java.io.FileInputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.StringWriter;

import java.util.SortedSet;
import java.util.TreeSet;
import java.util.Set;
import java.util.Vector;


public class UNLParserDebug implements UNLParserDebugConstants {

    /* Test program */
    public static void main(String args[]) throws ParseException, Exception {
        if (args.length != 1) throw new Exception("Please pass the file name as first and only arg.");
        UNLParserDebug parser =
            new UNLParserDebug(new BufferedReader(new InputStreamReader(new FileInputStream(args[0]))));

        parser.unlDocumentList();
    }

/*---------------------------------
 * UNL DOCUMENT STRUCTURE 
 *---------------------------------*/
  final public UNLDocument unlDocument() throws ParseException {
    Token docLabel;
    UNLDocument doc = null;
    UNLDocumentNode node;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case DOC:
      docLabel = jj_consume_token(DOC);
            doc = new UNLDocument(docLabel.image);
      break;
    default:
      jj_la1[0] = jj_gen;
      ;
    }
        // If there is no doc element, create it...
        if (null == doc) { doc = new UNLDocument("[D]"); } ;
    label_1:
    while (true) {
      node = structuredElement();
                doc.addElement(node);
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case PARAGRAPH:
      case TITLE:
      case SENTENCE:
      case END_PARAGRAPH:
        ;
        break;
      default:
        jj_la1[1] = jj_gen;
        break label_1;
      }
    }
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case END_DOC:
      jj_consume_token(END_DOC);
      break;
    default:
      jj_la1[2] = jj_gen;
      ;
    }
        {if (true) return doc;}
    throw new Error("Missing return statement in function");
  }

  final public UNLDocumentNode structuredElement() throws ParseException {
    Token t;
    UNLDocumentNode sent;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case PARAGRAPH:
      // Un document n'est pas structuré au niveau de la grammaire. 
          // Il s'agit simplement d'une succession d'éléments. Seule la phrase
          // et le titre sont définis.
              t = jj_consume_token(PARAGRAPH);
          {if (true) return new UNLDocumentNode(UNLDocumentNode.PARAGRAPH_START, t.image);}
      break;
    case END_PARAGRAPH:
      jj_consume_token(END_PARAGRAPH);
          {if (true) return new UNLDocumentNode(UNLDocumentNode.PARAGRAPH_END);}
      break;
    case TITLE:
      sent = title();
          {if (true) return sent;}
      break;
    case SENTENCE:
      sent = sentence();
          {if (true) return sent;}
      break;
    default:
      jj_la1[3] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
    throw new Error("Missing return statement in function");
  }

  final public UNLDocumentNode title() throws ParseException {
Graph g;
Token t;
    t = jj_consume_token(TITLE);
    g = relationList();
    jj_consume_token(END_TITLE);
        {if (true) return new UNLDocumentNode(UNLDocumentNode.TITLE, t.image, g);}
    throw new Error("Missing return statement in function");
  }

  final public UNLDocumentNode sentence() throws ParseException {
Graph g;
Token t;
    t = jj_consume_token(SENTENCE);
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case GENERIC_TOKEN:
      g = relationList();
      break;
    case NODELIST:
      g = nodeAndRelations();
      break;
    default:
      jj_la1[4] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
    jj_consume_token(END_SENTENCE);
        {if (true) return new UNLDocumentNode(UNLDocumentNode.SENTENCE, t.image, g);}
    throw new Error("Missing return statement in function");
  }

/*---------------------------------
 * RELATIONS 
 *---------------------------------*/
  final public Graph relationList() throws ParseException {
    GraphRelation rel;
    Graph g = new Graph();
    label_2:
    while (true) {
      rel = relation();
            g.addRelation(rel);
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case GENERIC_TOKEN:
        ;
        break;
      default:
        jj_la1[5] = jj_gen;
        break label_2;
      }
    }
        {if (true) return g;}
    throw new Error("Missing return statement in function");
  }

  final public GraphRelation relation() throws ParseException {
    Token rl;
    Token subGraphNumber = null;
    GraphNode n1, n2;
    rl = jj_consume_token(GENERIC_TOKEN);
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case COLON_NUMBER:
      subGraphNumber = jj_consume_token(COLON_NUMBER);
      break;
    default:
      jj_la1[6] = jj_gen;
      ;
    }
    jj_consume_token(PARO);
    n1 = node();
    jj_consume_token(COMMA);
    n2 = node();
    jj_consume_token(PARF);
        if (null == subGraphNumber)
            {if (true) return new GraphRelation(rl.image, n1, n2);}
        else
            {if (true) return new GraphRelation(rl.image, subGraphNumber.image, n1, n2);}
    throw new Error("Missing return statement in function");
  }

  final public GraphNode node() throws ParseException {
    UWNode uwn;
    SubGraphReferenceNode sgn;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case DBLQUOTED:
    case GENERIC_TOKEN:
      uwn = decoratedUniversalWord();
        {if (true) return (GraphNode) uwn;}
      break;
    case COLON_NUMBER:
      sgn = decoratedSubGraphNumber();
        {if (true) return (GraphNode) sgn;}
      break;
    default:
      jj_la1[7] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
    throw new Error("Missing return statement in function");
  }

  final public SubGraphReferenceNode decoratedSubGraphNumber() throws ParseException {
    Token num;
    Set attr;
    num = jj_consume_token(COLON_NUMBER);
    attr = attributes();
        {if (true) return new SubGraphReferenceNode(num.image, attr);}
    throw new Error("Missing return statement in function");
  }

/*---------------------------------
 * NODES AND RELATIONS  ([W] syntax)
 *---------------------------------*/
// For the moment, only accept isolated node.
  final public Graph nodeAndRelations() throws ParseException {
    GraphNode n;
    Graph g = new Graph();
    jj_consume_token(NODELIST);
    n = node();
    jj_consume_token(END_NODELIST);
        g.addNode(n);
        {if (true) return g;}
    throw new Error("Missing return statement in function");
  }

/*---------------------------------
 * DECORATED UW 
 *---------------------------------*/
  final public UWNode decoratedUniversalWord() throws ParseException {
    UniversalWord uw;
    Token num = null;
    Set attr;
    uw = universalWord();
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case COLON_NUMBER:
      num = jj_consume_token(COLON_NUMBER);
      break;
    default:
      jj_la1[8] = jj_gen;
      ;
    }
    attr = attributes();
    if (null == num) {
        {if (true) return new UWNode(uw, attr);}
    } else {
        {if (true) return new UWNode(uw, num.image, attr);}
    }
    throw new Error("Missing return statement in function");
  }

  final public Set attributes() throws ParseException {
    Set attr = (Set) new TreeSet();
    Token a;
    label_3:
    while (true) {
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case ATTR:
        ;
        break;
      default:
        jj_la1[9] = jj_gen;
        break label_3;
      }
      a = jj_consume_token(ATTR);
            attr.add(a.image);
    }
        {if (true) return attr;}
    throw new Error("Missing return statement in function");
  }

/*---------------------------------
 * UNIVERSAL WORD 
 *---------------------------------*/
  final public UniversalWord universalWord() throws ParseException {
String hw;
SortedSet restrictions;
    hw = headword();
    restrictions = restrictionList();
        {if (true) return new UniversalWord(hw, restrictions);}
    throw new Error("Missing return statement in function");
  }

  final public String headword() throws ParseException {
    Token t;
    StringBuffer hw = new StringBuffer();
    label_4:
    while (true) {
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case GENERIC_TOKEN:
        t = jj_consume_token(GENERIC_TOKEN);
            hw.append(t.image);
            hw.append(" ");
        break;
      case DBLQUOTED:
        t = jj_consume_token(DBLQUOTED);
            hw.append(t.image);
            hw.append(" ");
        break;
      default:
        jj_la1[10] = jj_gen;
        jj_consume_token(-1);
        throw new ParseException();
      }
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case DBLQUOTED:
      case GENERIC_TOKEN:
        ;
        break;
      default:
        jj_la1[11] = jj_gen;
        break label_4;
      }
    }
        int last = hw.length()-1;
        if ((-1 != last) && (hw.charAt(last) == ' ')) {
            hw.setLength(last);
        }
        {if (true) return hw.toString();}
    throw new Error("Missing return statement in function");
  }

  final public SortedSet restrictionList() throws ParseException {
    Restriction r;
    SortedSet restrictions = (SortedSet) new TreeSet();
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case PARO:
      jj_consume_token(PARO);
      r = restriction();
          restrictions.add(r);
      label_5:
      while (true) {
        switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
        case COMMA:
          ;
          break;
        default:
          jj_la1[12] = jj_gen;
          break label_5;
        }
        jj_consume_token(COMMA);
        r = restriction();
              restrictions.add(r);
      }
      jj_consume_token(PARF);
      break;
    default:
      jj_la1[13] = jj_gen;
      ;
    }
        {if (true) return restrictions;}
    throw new Error("Missing return statement in function");
  }

  final public Restriction restriction() throws ParseException {
    Token label;
    Token direction;
    UniversalWord uw;
    label = jj_consume_token(GENERIC_TOKEN);
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case CHEVRD:
      direction = jj_consume_token(CHEVRD);
      break;
    case CHEVRG:
      direction = jj_consume_token(CHEVRG);
      break;
    default:
      jj_la1[14] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
    uw = embeddedUniversalWord(label.image);
        {if (true) return new Restriction(label.image, direction.image.charAt(0), uw);}
    throw new Error("Missing return statement in function");
  }

/* Putain de syntaxe à la con ! */
/* Un UW dans une restriction n'a pas forcément la même syntaxe qu'une UW
   "normale". En effet, la restricition peut être factorisée. */
  final public UniversalWord embeddedUniversalWord(String inheritedLabel) throws ParseException {
    String hw;
    SortedSet restrictions;
    hw = headword();
    restrictions = embeddedRestrictionList(inheritedLabel);
        {if (true) return new UniversalWord(hw, restrictions);}
    throw new Error("Missing return statement in function");
  }

  final public SortedSet embeddedRestrictionList(String inheritedLabel) throws ParseException {
    UniversalWord uw;
    SortedSet restrictions;
    Token direction;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case CHEVRD:
    case CHEVRG:
            restrictions = (SortedSet) new TreeSet();
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case CHEVRD:
        direction = jj_consume_token(CHEVRD);
        break;
      case CHEVRG:
        direction = jj_consume_token(CHEVRG);
        break;
      default:
        jj_la1[15] = jj_gen;
        jj_consume_token(-1);
        throw new ParseException();
      }
      uw = embeddedUniversalWord(inheritedLabel);
            restrictions.add(new Restriction(inheritedLabel, direction.image.charAt(0), uw));
      break;
    default:
      jj_la1[16] = jj_gen;
      restrictions = restrictionList();
    }
        {if (true) return restrictions;}
    throw new Error("Missing return statement in function");
  }

/*---------------------------------
 * RULES FOR TESTS 
 *---------------------------------*/

/**
 * List of universal words separated by commas.
 */
  final public void universalWordList() throws ParseException {
    decoratedUniversalWord();
        System.out.println("UW");
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case COMMA:
      jj_consume_token(COMMA);
      universalWordList();
      break;
    default:
      jj_la1[17] = jj_gen;

    }
    jj_consume_token(0);
  }

/**
 * Simple unl graph.
 */
  final public void isolatedUnlGraph() throws ParseException {
    relationList();
    jj_consume_token(0);
  }

/**
 * Main entry point: unl document List.
 */
  final public Vector unlDocumentList() throws ParseException {
UNLDocument doc;
Vector list = new Vector();
    label_6:
    while (true) {
      doc = unlDocument();
           list.add(doc);
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case DOC:
      case PARAGRAPH:
      case TITLE:
      case SENTENCE:
      case END_PARAGRAPH:
        ;
        break;
      default:
        jj_la1[18] = jj_gen;
        break label_6;
      }
    }
        {if (true) return list;}
    throw new Error("Missing return statement in function");
  }

  public UNLParserDebugTokenManager token_source;
  SimpleCharStream jj_input_stream;
  public Token token, jj_nt;
  private int jj_ntk;
  private int jj_gen;
  final private int[] jj_la1 = new int[19];
  final private int[] jj_la1_0 = {0x10000,0x4e0000,0x200000,0x4e0000,0x100000,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x8000,0x800,0x6000,0x6000,0x6000,0x8000,0x4f0000,};
  final private int[] jj_la1_1 = {0x0,0x0,0x0,0x0,0x8,0x8,0x2,0xe,0x2,0x10,0xc,0xc,0x0,0x0,0x0,0x0,0x0,0x0,0x0,};

  public UNLParserDebug(java.io.InputStream stream) {
    jj_input_stream = new SimpleCharStream(stream, 1, 1);
    token_source = new UNLParserDebugTokenManager(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 19; i++) jj_la1[i] = -1;
  }

  public void ReInit(java.io.InputStream stream) {
    jj_input_stream.ReInit(stream, 1, 1);
    token_source.ReInit(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 19; i++) jj_la1[i] = -1;
  }

  public UNLParserDebug(java.io.Reader stream) {
    jj_input_stream = new SimpleCharStream(stream, 1, 1);
    token_source = new UNLParserDebugTokenManager(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 19; i++) jj_la1[i] = -1;
  }

  public void ReInit(java.io.Reader stream) {
    jj_input_stream.ReInit(stream, 1, 1);
    token_source.ReInit(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 19; i++) jj_la1[i] = -1;
  }

  public UNLParserDebug(UNLParserDebugTokenManager tm) {
    token_source = tm;
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 19; i++) jj_la1[i] = -1;
  }

  public void ReInit(UNLParserDebugTokenManager tm) {
    token_source = tm;
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 19; i++) jj_la1[i] = -1;
  }

  final private Token jj_consume_token(int kind) throws ParseException {
    Token oldToken;
    if ((oldToken = token).next != null) token = token.next;
    else token = token.next = token_source.getNextToken();
    jj_ntk = -1;
    if (token.kind == kind) {
      jj_gen++;
      return token;
    }
    token = oldToken;
    jj_kind = kind;
    throw generateParseException();
  }

  final public Token getNextToken() {
    if (token.next != null) token = token.next;
    else token = token.next = token_source.getNextToken();
    jj_ntk = -1;
    jj_gen++;
    return token;
  }

  final public Token getToken(int index) {
    Token t = token;
    for (int i = 0; i < index; i++) {
      if (t.next != null) t = t.next;
      else t = t.next = token_source.getNextToken();
    }
    return t;
  }

  final private int jj_ntk() {
    if ((jj_nt=token.next) == null)
      return (jj_ntk = (token.next=token_source.getNextToken()).kind);
    else
      return (jj_ntk = jj_nt.kind);
  }

  private java.util.Vector jj_expentries = new java.util.Vector();
  private int[] jj_expentry;
  private int jj_kind = -1;

  final public ParseException generateParseException() {
    jj_expentries.removeAllElements();
    boolean[] la1tokens = new boolean[37];
    for (int i = 0; i < 37; i++) {
      la1tokens[i] = false;
    }
    if (jj_kind >= 0) {
      la1tokens[jj_kind] = true;
      jj_kind = -1;
    }
    for (int i = 0; i < 19; i++) {
      if (jj_la1[i] == jj_gen) {
        for (int j = 0; j < 32; j++) {
          if ((jj_la1_0[i] & (1<<j)) != 0) {
            la1tokens[j] = true;
          }
          if ((jj_la1_1[i] & (1<<j)) != 0) {
            la1tokens[32+j] = true;
          }
        }
      }
    }
    for (int i = 0; i < 37; i++) {
      if (la1tokens[i]) {
        jj_expentry = new int[1];
        jj_expentry[0] = i;
        jj_expentries.addElement(jj_expentry);
      }
    }
    int[][] exptokseq = new int[jj_expentries.size()][];
    for (int i = 0; i < jj_expentries.size(); i++) {
      exptokseq[i] = (int[])jj_expentries.elementAt(i);
    }
    return new ParseException(token, exptokseq, tokenImage);
  }

  final public void enable_tracing() {
  }

  final public void disable_tracing() {
  }

}
