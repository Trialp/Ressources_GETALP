//
//  QueryCriteria.java
//  xcode_Papillon5.1.lexalp
//
//  Created by Francis Brunet-Manquat on 10/11/05.
//  Copyright 2005 __MyCompanyName__. All rights reserved.
//

package fr.imag.clips.papillon.business.dictionary;

/* standards imports */

import com.lutris.dods.builder.generator.query.QueryBuilder;

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class QueryCriteria {

    private ArrayList criteriaList;
	private String key= null;
	private String lang= null;
	private String clause= null;
    
    public static final String EQUAL = "0";
    public static final String NOT_EQUAL = "1";
    public static final String CASE_INSENSITIVE_EQUAL = "2";
    public static final String CASE_INSENSITIVE_NOT_EQUAL = "3";
    public static final String CASE_INSENSITIVE_CONTAINS = "4";
    public static final String CASE_INSENSITIVE_ENDS_WITH = "5";
    public static final String CASE_INSENSITIVE_STARTS_WITH = "6";
    public static final String CASE_SENSITIVE_CONTAINS = "7";
    public static final String CASE_SENSITIVE_ENDS_WITH = "8";
    public static final String CASE_SENSITIVE_STARTS_WITH = "9";
    public static final String GREATER_THAN = "10";
    public static final String LESS_THAN = "11";
    
    public QueryCriteria() {
        criteriaList = new ArrayList();
    }
	
	/*
	key[0] = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_FACET + "." + Integer.toString(i));
            key[1] = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_SOURCE + "." + Integer.toString(i));
            String facetValue = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_FACETVALUE + "." + Integer.toString(i));
            key[2] = (null == facetValue) ? "" : facetValue;
            String comparator = AbstractPO.myGetParameter(req, AdvancedQueryFormXHTML.NAME_OPERATOR + "." + Integer.toString(i));
            //int comparisonOperator = (comparator != null) ? Integer.valueOf(comparator).intValue() : 0;
            //key[3] = IQuery.QueryBuilderStrategy[comparisonOperator];
            key[3] = comparator;
*/
    
    // find words in string value to add into query : column=word[0] AND column=word[1] AND ...
    public void addAdvancedValue(String column, String strategie, String value) {
        
        // find words in string value
        Pattern pat = Pattern.compile("^[ ]*([^ ]+)(.*)$");
        Matcher mat = pat.matcher(value);
        boolean isFind = mat.find();
        //System.out.println("addAdvancedValue: $" + value + "$");
        
        while (isFind) {
            
            // add word in criteria
            add(column, strategie, mat.group(1));
            
            //
            if ( (mat.group(2) != null) || (!mat.group(2).equals("")) ) {
                value = mat.group(2);
                mat = pat.matcher(value);
                isFind = mat.find();
            } else {
                isFind = false;
            }
        }
    }
    
    //
    public void add(String column, String strategie, String value) {
        
		if (column.equals("key")) {
			this.key=value;
		}
		else if (column.equals("lang")) {
			this.lang=value;
		}
		
        //
        Pattern pat = Pattern.compile("^[ ]*([^ ]+([ ]+[^ ]+)*)[ ]*$");
        Matcher mat = pat.matcher(value);
        if (mat.find()) value = mat.group(1);
        //System.out.println("add: $" + value + "$");
        
        //
        String[] criteria = new String[3];
        criteria[0] = column;
        if (strategie==null || strategie.equals("")) {
			strategie = QueryCriteria.EQUAL;
		}
        if ( strategie.equals(QueryCriteria.EQUAL) ) {
            criteria[1] = QueryBuilder.EQUAL;
            criteria[2] = value; 
            
        } else if ( strategie.equals(QueryCriteria.NOT_EQUAL) ) {
            criteria[1] = QueryBuilder.NOT_EQUAL;
            criteria[2] = value; 
            
        } else if ( strategie.equals(QueryCriteria.CASE_INSENSITIVE_EQUAL) ) {
            criteria[1] = "ilike";
            criteria[2] = value; 
            
        } else if ( strategie.equals(QueryCriteria.CASE_INSENSITIVE_NOT_EQUAL) ) {
            criteria[1] = "not ilike";
            criteria[2] = value; 
            
        } else if ( strategie.equals(QueryCriteria.CASE_INSENSITIVE_CONTAINS) ) {
            criteria[1] = "ilike";
            criteria[2] = "%" + value + "%"; 
        
        } else if ( strategie.equals(QueryCriteria.CASE_INSENSITIVE_ENDS_WITH) ) {
            criteria[1] = "ilike";
            criteria[2] = "%" + value; 
        
        }  else if ( strategie.equals(QueryCriteria.CASE_INSENSITIVE_STARTS_WITH) ) {
            criteria[1] = "ilike";
            criteria[2] = value + "%"; 
            
        }  else if ( strategie.equals(QueryCriteria.CASE_SENSITIVE_CONTAINS) ) {
            criteria[1] = "like";
            criteria[2] = "%" + value + "%"; 
            
        } else if ( strategie.equals(QueryCriteria.CASE_SENSITIVE_ENDS_WITH) ) {
            criteria[1] = "like";
            criteria[2] = "%" + value; 
            
        }  else if ( strategie.equals(QueryCriteria.CASE_SENSITIVE_STARTS_WITH) ) {
            criteria[1] = "like";
            criteria[2] = value + "%"; 
        
        }  else if ( strategie.equals(QueryCriteria.GREATER_THAN) ) {
            criteria[1] = ">";
            criteria[2] = value; 
            
        }  else if ( strategie.equals(QueryCriteria.LESS_THAN) ) {
            criteria[1] = "<";
            criteria[2] = value; 
                    
        } else {
            // defaut
            criteria[1] = strategie;
            criteria[2] = value;
        }
		if (!column.equals("key") && !column.equals("lang"))  {
			addClause(column + " " + criteria[1] + " '" + criteria[2] + "'");		
		}
		
        criteriaList.add(criteria);
    }
    
    //
    public int size() {
        return criteriaList.size();
    }
    
    // 
    public String getColumn(int i) {
        return ((String[])criteriaList.get(i))[0];
    }
    
    // 
    public String getStrategie(int i) {
        return ((String[])criteriaList.get(i))[1];
    }
    
    // 
    public String getValue(int i) {
        return ((String[])criteriaList.get(i))[2];
    }
    
    // 
    public int getCriteriaByColumn(String column) {
        
        int i = 0;
        while ( !((String[])criteriaList.get(i))[0].equals(column) && (i < criteriaList.size()) ) {
            i++;
        }
        
        return i;
    }
    
	public String getLang() {
		return lang;
	}
    
	public String getKey() {
		return key;
	}
    
	public String getClause() {
		return clause;
	}
	public String getFullClause() {
		String resClause = "";
		if (this.key != null) {
			resClause += "key='"+this.key+"'";
		}
		if (this.lang != null) {
			if (this.key != null) {
			resClause += " AND ";
			}
			resClause += "lang='"+this.lang+"'";
		}
		if (this.clause != null && !resClause.equals("")) {
            resClause = this.clause + " AND " + resClause;
		}
        else if (this.clause != null) {
            resClause = this.clause;
        }
		return resClause;
	}
	
	// following BUG311, needed to create clauses
    // needed also to set the method public
	public void addClause(String theClause) {
		if (this.clause != null) {
			this.clause+= " AND " + theClause;		
		}
		else {
			this.clause = theClause;		
		}
	}

	// following BUG311, needed to create clauses
	public void appendClause(String theClause) {
		this.clause = "(" + this.clause + theClause + ")";
	}
    
    // 
    public void remove(int i) {
        criteriaList.remove(i);
    }
}
