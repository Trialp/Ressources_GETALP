<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0">
    <xsl:output method="text" encoding="iso-8859-1"/>
    <xsl:template name="fileFooter11">
<xsl:value-of select="COMMIT_REFERENCED_DOS"/> <!-- CommitDeleteOtherDO -->
                    // WebDocWf extension for save without references
                    // The following line has been inserted
                }
                // end of WebDocWf extension for save without references
            }
            if ( false ) {
                // This throw is here to keep the compiler happy
                // in the case of a DO that does not refer to other DOs.
                // In that case, the above delete/commit code blocks will be empty
                // and throw nothing.
                throw new DataObjectException( "foo" );   
            }
            if ( delete ) {
              dbt.delete( this );
              setDeleted(true);
              if (isRootDeleteNode) {
                   _tr_(dbt).resetDeletedDOs();
              }
            } else {
                if ( isLoaded() || isAutoSaveCreateVirgin())
                    dbt.insert( this );   // dbt.insert() handles insertions and updates
            }
            if (ownTransaction) {
                dbt.commit(); // commit the transaction
            }
        } catch (SQLException sqle) {
            StringBuffer message = new StringBuffer("Failed to insert/update DO: ");
            message.append(sqle.getMessage());
            // rollback, if necessary
            if (ownTransaction) {
                try {
                    dbt.rollback();
                } catch (SQLException sqle2) {
                    message.insert(0,"\n");
                    message.insert(0,sqle2.getMessage());
                    message.insert(0,"Rollback failed: ");
                }
            }
            throw new SQLException(message.toString());
        } finally {
            // release the transaction, if any
            if (ownTransaction) {
                dbt.release();
            }
        }
    }

    private void changedFlags_set(boolean value) {
</xsl:template>
</xsl:stylesheet>
