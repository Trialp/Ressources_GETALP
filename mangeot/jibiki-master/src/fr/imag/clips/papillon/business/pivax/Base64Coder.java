package fr.imag.clips.papillon.business.pivax;

/**
* Hong-Thai Nguyen
* 22/03/2010
* A Base64 encoder/decoder for Q-System basing on Base64 standard
* <p>
* This class is used to encode and decode data in Base64 format as described in RFC 1521.
*
* <p>
* Replace '=' by '_' in order to be accepted in Q-Systems format
* 
*/
public class Base64Coder {

// The line separator string of the operating system.
private static final String systemLineSeparator = "\n";
private static final char LEGAL_EQUAL_CARACTER = '=';
private static final char PLUS_EQUAL_CARACTER = '+';
private static final char SLASH_EQUAL_CARACTER = '/';

// Mapping table from 6-bit nibbles to Base64 characters.
private static char[]    map1 = new char[64];
   static {
      int i=0;
      for (char c='A'; c<='Z'; c++) map1[i++] = c;
      for (char c='a'; c<='z'; c++) map1[i++] = c;
      for (char c='0'; c<='9'; c++) map1[i++] = c;
      map1[i++] = '+'; map1[i++] = '/'; }

// Mapping table from Base64 characters to 6-bit nibbles.
private static byte[]    map2 = new byte[128];
   static {
      for (int i=0; i<map2.length; i++) map2[i] = -1;
      for (int i=0; i<64; i++) map2[map1[i]] = (byte)i; }

/**
* Encodes a string into Base64 format.
* No blanks or line breaks are inserted.
* @param s  A String to be encoded.
* @return   A String containing the Base64 encoded data.
*/
public static String encodeString (String s) {
   return new String(encode(s.getBytes())); }

/**
* Encodes a byte array into Base64 format. replace non-valided SysQ format
* No blanks or line breaks are inserted in the output.
* @param in  An array containing the data bytes to be encoded.
* @return    A character array containing the Base64 encoded data.
*/
public static String encodeSysQ (String in) {
	
  String out = encodeString(in);
  
  // java.lang.System.out.println("encodeSysQ before " + out);
  
  int pos = out.indexOf("=");
  while (pos >= 0) {
	  out = out.substring(0, pos) + "equal" + out.substring(pos+1, out.length());
	  pos = out.indexOf("=", pos+1);
  }	  
  pos = out.indexOf("+");
  while (pos >= 0) {
	  out = out.substring(0, pos) + "plus" + out.substring(pos+1, out.length());
	  pos = out.indexOf("+", pos+1);
  }	
  
  pos = out.indexOf("/");
  while (pos >= 0) {
	  out = out.substring(0, pos) + "slash" + out.substring(pos+1, out.length());
	  pos = out.indexOf("/", pos+1);
  }	
  
  // java.lang.System.out.println("encodeSysQ after " + out);
   return out;
 }

/**
* Decodes a byte array into Base64 format. replace non-valided SysQ format
* No blanks or line breaks are inserted in the output.
* @param in  An array containing the data bytes to be encoded.
* @return    A character array containing the Base64 encoded data.
*/
public static String decodeSysQ (String in) {
  String out = in;
  // java.lang.System.out.println("decodeSysQ before " + out);
  
  int pos = out.indexOf("equal");
  while (pos >= 0) {
	 out = out.substring(0, pos) + "=" + out.substring(pos+"equal".length(), out.length());
	 pos = out.indexOf("equal", pos+1);
  }
  
  pos = out.indexOf("plus"); 
  while (pos >= 0) {
	  out = out.substring(0, pos) + "+" + out.substring(pos+"plus".length(), out.length());
	  pos = out.indexOf("plus", pos+1);
  }  
  
  pos = out.indexOf("slash"); 
  while (pos >= 0) {
	  out = out.substring(0, pos) + "/" + out.substring(pos+"slash".length(), out.length());
	  pos = out.indexOf("slash", pos+1);
  }  
  
  // java.lang.System.out.println("decodeSysQ after " + out);
  
  out = decodeString(out);
  return out;
 }

/**
* Encodes a byte array into Base64 format.
* No blanks or line breaks are inserted in the output.
* @param in  An array containing the data bytes to be encoded.
* @return    A character array containing the Base64 encoded data.
*/
public static char[] encode (byte[] in) {
   return encode(in, 0, in.length); }

/**
* Encodes a byte array into Base64 format.
* No blanks or line breaks are inserted in the output.
* @param in    An array containing the data bytes to be encoded.
* @param iLen  Number of bytes to process in <code>in</code>.
* @return      A character array containing the Base64 encoded data.
*/
public static char[] encode (byte[] in, int iLen) {
   return encode(in, 0, iLen); }

/**
* Encodes a byte array into Base64 format.
* No blanks or line breaks are inserted in the output.
* @param in    An array containing the data bytes to be encoded.
* @param iOff  Offset of the first byte in <code>in</code> to be processed.
* @param iLen  Number of bytes to process in <code>in</code>, starting at <code>iOff</code>.
* @return      A character array containing the Base64 encoded data.
*/
public static char[] encode (byte[] in, int iOff, int iLen) {
   int oDataLen = (iLen*4+2)/3;       // output length without padding
   int oLen = ((iLen+2)/3)*4;         // output length including padding
   char[] out = new char[oLen];
   int ip = iOff;
   int iEnd = iOff + iLen;
   int op = 0;
   while (ip < iEnd) {
      int i0 = in[ip++] & 0xff;
      int i1 = ip < iEnd ? in[ip++] & 0xff : 0;
      int i2 = ip < iEnd ? in[ip++] & 0xff : 0;
      int o0 = i0 >>> 2;
      int o1 = ((i0 &   3) << 4) | (i1 >>> 4);
      int o2 = ((i1 & 0xf) << 2) | (i2 >>> 6);
      int o3 = i2 & 0x3F;
      out[op++] = map1[o0];
      out[op++] = map1[o1];
      // out[op] = op < oDataLen ? map1[o2] : '='; op++;
      out[op] = op < oDataLen ? map1[o2] : LEGAL_EQUAL_CARACTER; op++;
      // out[op] = op < oDataLen ? map1[o3] : '='; op++; }
      out[op] = op < oDataLen ? map1[o3] : LEGAL_EQUAL_CARACTER; op++; }
   return out; }

/**
* Decodes a string from Base64 format.
* No blanks or line breaks are allowed within the Base64 encoded input data.
* @param s  A Base64 String to be decoded.
* @return   A String containing the decoded data.
* @throws   IllegalArgumentException If the input is not valid Base64 encoded data.
*/
public static String decodeString (String s) {
   return new String(decode(s)); }

/**
* Decodes a byte array from Base64 format and ignores line separators, tabs and blanks.
* CR, LF, Tab and Space characters are ignored in the input data.
* This method is compatible with <code>sun.misc.BASE64Decoder.decodeBuffer(String)</code>.
* @param s  A Base64 String to be decoded.
* @return   An array containing the decoded data bytes.
* @throws   IllegalArgumentException If the input is not valid Base64 encoded data.
*/
public static byte[] decodeLines (String s) {
   char[] buf = new char[s.length()];
   int p = 0;
   for (int ip = 0; ip < s.length(); ip++) {
      char c = s.charAt(ip);
      if (c != ' ' && c != '\r' && c != '\n' && c != '\t')
         buf[p++] = c; }
   return decode(buf, 0, p); }

/**
* Decodes a byte array from Base64 format.
* No blanks or line breaks are allowed within the Base64 encoded input data.
* @param s  A Base64 String to be decoded.
* @return   An array containing the decoded data bytes.
* @throws   IllegalArgumentException If the input is not valid Base64 encoded data.
*/
public static byte[] decode (String s) {
   return decode(s.toCharArray()); }

/**
* Decodes a byte array from Base64 format.
* No blanks or line breaks are allowed within the Base64 encoded input data.
* @param in  A character array containing the Base64 encoded data.
* @return    An array containing the decoded data bytes.
* @throws    IllegalArgumentException If the input is not valid Base64 encoded data.
*/
public static byte[] decode (char[] in) {
   return decode(in, 0, in.length); }

/**
* Decodes a byte array from Base64 format.
* No blanks or line breaks are allowed within the Base64 encoded input data.
* @param in    A character array containing the Base64 encoded data.
* @param iOff  Offset of the first character in <code>in</code> to be processed.
* @param iLen  Number of characters to process in <code>in</code>, starting at <code>iOff</code>.
* @return      An array containing the decoded data bytes.
* @throws      IllegalArgumentException If the input is not valid Base64 encoded data.
*/
public static byte[] decode (char[] in, int iOff, int iLen) {
   if (iLen%4 != 0) throw new IllegalArgumentException ("Length of Base64 encoded input string is not a multiple of 4.");
   while (iLen > 0 && in[iOff+iLen-1] == LEGAL_EQUAL_CARACTER) iLen--;
   int oLen = (iLen*3) / 4;
   byte[] out = new byte[oLen];
   int ip = iOff;
   int iEnd = iOff + iLen;
   int op = 0;
   while (ip < iEnd) {
      int i0 = in[ip++];
      int i1 = in[ip++];
      int i2 = ip < iEnd ? in[ip++] : 'A';
      int i3 = ip < iEnd ? in[ip++] : 'A';
      if (i0 > 127 || i1 > 127 || i2 > 127 || i3 > 127)
         throw new IllegalArgumentException ("Illegal character in Base64 encoded data.");
      int b0 = map2[i0];
      int b1 = map2[i1];
      int b2 = map2[i2];
      int b3 = map2[i3];
      if (b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0)
         throw new IllegalArgumentException ("Illegal character in Base64 encoded data.");
      int o0 = ( b0       <<2) | (b1>>>4);
      int o1 = ((b1 & 0xf)<<4) | (b2>>>2);
      int o2 = ((b2 &   3)<<6) |  b3;
      out[op++] = (byte)o0;
      if (op<oLen) out[op++] = (byte)o1;
      if (op<oLen) out[op++] = (byte)o2; }
   return out; }

// Dummy constructor.
private Base64Coder() {}

} // end class Base64Coder