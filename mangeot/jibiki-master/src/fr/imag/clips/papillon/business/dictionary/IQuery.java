/*
 -----------------------------
 * $Id$
 *-----------------------------------------------
 * $Log$
 * Revision 1.4  2005/07/21 15:09:20  mangeot
 * Bug fixes and corrections mainly for the GDEF project
 *
 * Revision 1.3  2005/06/15 16:48:27  mangeot
 * Merge between the ContribsInXml branch and the main trunk. It compiles but bugs remain..
 *
 * Revision 1.2.4.1  2005/04/29 14:50:25  mangeot
 * New version with contribution infos embedded in the XML of the entries
 *
 *
 *-----------------------------------------------
 */ 

package fr.imag.clips.papillon.business.dictionary;

import com.lutris.dods.builder.generator.query.QueryBuilder;

public interface IQuery {
/**
 * Insert the method's description here.
 * Creation date: (10.08.01 20:55:40)
 * @return org.dict.IDatabase
 */
    public static final int STRATEGY_NONE = -1;
    // exact "Match words exactly"
    public static final int STRATEGY_EXACT = 0;
    public static final int STRATEGY_EQUAL = 0;

    // prefix "Match prefixes"
    public static final int STRATEGY_PREFIX = 1;

    // suffix "Match suffixes"
    public static final int STRATEGY_SUFFIX = 2;

    // substring "Match substring occurring anywhere in word"
    public static final int STRATEGY_SUBSTRING = 3;

    // insensitive "Match words case-insensitively"
	public static final int STRATEGY_INSENSITIVE_EXACT = 4;

    // inprefix "Match case-insensitive prefixes"
    public static final int STRATEGY_INSENSITIVE_PREFIX = 5;

    // insuffix "Match case-insensitive suffixes"
    public static final int STRATEGY_INSENSITIVE_SUFFIX = 6;

    // insubstring "Match case-insensitive substring occurring anywhere in word"
    public static final int STRATEGY_INSENSITIVE_SUBSTRING = 7;

    // lemmatize "Using XRCE lemmatizer module, for eng, fra, esp, ita"
    public static final int STRATEGY_LEMMATIZE = 8;
	
    // foks "Using Slaven BILAC's FOKS module, only for Japanese"
    public static final int STRATEGY_FOKS = 9;
	
    // !=
    public static final int STRATEGY_NOT_EQUAL = 10;

    // >
    public static final int STRATEGY_GREATER_THAN = 11;
	
    // >=
    public static final int STRATEGY_GREATER_THAN_OR_EQUAL = 12;
	
    // >
    public static final int STRATEGY_LESS_THAN = 13;
	
    // >=
    public static final int STRATEGY_LESS_THAN_OR_EQUAL = 14;
	
	public final static String [] QueryBuilderStrategy = { 
		QueryBuilder.EQUAL, 
		QueryBuilder.EQUAL, 
		QueryBuilder.CASE_SENSITIVE_STARTS_WITH, 
		QueryBuilder.CASE_SENSITIVE_ENDS_WITH, 
		QueryBuilder.CASE_SENSITIVE_CONTAINS, 
		QueryBuilder.CASE_INSENSITIVE_EQUAL, 
		QueryBuilder.CASE_INSENSITIVE_STARTS_WITH, 
		QueryBuilder.CASE_INSENSITIVE_ENDS_WITH, 
		QueryBuilder.CASE_INSENSITIVE_CONTAINS, 
		QueryBuilder.EQUAL, 
		QueryBuilder.EQUAL, 
		QueryBuilder.NOT_EQUAL, 
		QueryBuilder.GREATER_THAN, 
		QueryBuilder.GREATER_THAN_OR_EQUAL, 
		QueryBuilder.LESS_THAN, 
		QueryBuilder.LESS_THAN_OR_EQUAL, 
		};
}
