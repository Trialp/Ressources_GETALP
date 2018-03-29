/* 
 * Papillon 2001
 * © Mathieu Mangeot & Gilles Sérasset, GETA CLIPS IMAG
 * -------------------------------------------------------------
 * $Id$
 * -------------------------------------------------------------
 * $Log$
 * Revision 1.2  2005/01/15 12:51:24  mangeot
 * Deleting old cvs comments + bug fixes with xhtml and enhydra5.1
 *
 * Revision 1.1.1.1  2004/12/06 16:38:54  serasset
 * Papillon for enhydra 5.1. This version compiles and starts with enhydra 5.1.
 * There are still bugs in the code.
 *
 * -------------------------------------------------------------
 * This SQL creates the user papillon.
 */

GRANT ALL ON messages TO papillon;
GRANT ALL ON objectid TO papillon;
GRANT ALL ON xslsheets TO papillon;
GRANT ALL ON informationfiles TO papillon;
GRANT ALL ON informationdocument TO papillon;
GRANT ALL ON users TO papillon;
GRANT ALL ON history TO papillon;
GRANT ALL ON dictionaries TO papillon;
GRANT ALL ON volumes TO papillon;

GRANT ALL ON papillonaxi TO papillon;
