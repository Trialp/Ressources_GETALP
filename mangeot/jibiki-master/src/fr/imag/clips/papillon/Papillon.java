/*
 * papillon 
 *
 * Enhydra super-servlet
 * 
 * © Mathieu Mangeot & Gilles Sérasset - GETA CLIPS IMAG
 * Projet Papillon
 *-----------------------------------------------
 * $Id$
 *-----------------------------------------------
*-----------------------------------------------
 * Papillon enhydra application.
 */

package fr.imag.clips.papillon;

import com.lutris.appserver.server.*;
import com.lutris.appserver.server.httpPresentation.*;
import com.lutris.util.*;
import com.lutris.logging.Logger;

import fr.imag.clips.papillon.business.dictionary.DictionariesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;
import fr.imag.clips.papillon.business.dictionary.AvailableLanguages;
import fr.imag.clips.papillon.business.xsl.XslSheetFactory;
import fr.imag.clips.papillon.business.PapillonBusinessException;
import fr.imag.clips.papillon.business.PapillonLogger;

/* For JDictd */
import fr.imag.clips.papillon.dict.server.JDictd;

/*for tests*/
import fr.imag.clips.papillon.business.xml.XMLParsersPool;
import fr.imag.clips.papillon.papillon_data.DataLayerVersion;


/**
 * The application object.
 *
 * Application-wide data would go here.
 */
public class Papillon extends StandardApplication {
	
    protected static final String PRIORITY_PACKAGE_CONFIG = "Papillon.Presentation.PriorityPackage";
    protected static final String LAYOUT_CONFIG = "Papillon.LayoutClassName";
    protected static final String COOKIE_CONFIG = "Papillon.LoginCookieName";
    protected static final String PAGE_EXPIRE_TIME = "Papillon.LoginCookieName";

    protected String presentationPriorityPackage = null;
    protected String layoutClassName = "fr.imag.clips.papillon.presentation.PapillonLayout";
    protected String loginCookieName = "PapillonLoginCookie";
	protected String applicationPrefix = "/";
    protected String appName = "Papillon";
    
        
    /* Expiration du cache dans une semaine */
    /* protected int pageExpireTime = 7 * 24 * 60 * 60;
    Finalement, désactivation du cache */
    protected int pageExpireTime = 0;
	static {
                System.setProperty("javax.xml.parsers.DocumentBuilderFactory", 
				"org.apache.xerces.jaxp.DocumentBuilderFactoryImpl");
        }
    /*
     *  A few methods you might want to add to.
     *  See StandardApplication for more details.
     */
    public void startup(Config appConfig) throws ApplicationException {
        super.startup(appConfig);
        //  Here is where you would read application-specific settings from
        //  your config file.
        
        appName = Enhydra.getApplication().getConfig().getConfigFile().getFile().getName();
        if (appName.indexOf(".") > 0) {
            appName = appName.substring(0, appName.lastIndexOf("."));
        }
        
        // Look at the Xerces version that is currently loaded and display it...
        PapillonLogger.writeInfoMsg(org.apache.xerces.impl.Version.getVersion());
		
		// Look at the local file encoding
        PapillonLogger.writeInfoMsg("Local system file encoding: "+System.getProperty ("file.encoding"));
		// Available only in java 1.5
		//java.nio.charset.Charset defaultCharset = java.nio.charset.Charset.defaultCharset();
		java.io.InputStreamReader myInputStreamReader = new java.io.InputStreamReader(System.in);
        PapillonLogger.writeInfoMsg("Local system default charset: "+myInputStreamReader.getEncoding());
        //  Dictd specific settings.
        boolean listen = false; 
        try {
            listen =  Enhydra.getApplication().getConfig().getBoolean(JDictd.LISTEN_STRING);
        } catch (ConfigException e) {
            throw new ApplicationException("Dictd parameters not found. Check the application config file.", e);
        }
        if (listen) {
            
            int port = 2628;
            int timeout = 50000;
            try {
                port =  Enhydra.getApplication().getConfig().getInt(JDictd.PORT_STRING);
                timeout =  Enhydra.getApplication().getConfig().getInt(JDictd.TIMEOUT_STRING);
            }
            catch (ConfigException e) {
                throw new ApplicationException("Dictd parameters not found. Check the application config file.", e);
            }
            // Here is where I startup JDictd server
            JDictd.listen(port,Enhydra.getApplication(),timeout);
        }

        // Initialize the XMLParsersPool
        try {
            XMLParsersPool.initializeXMLParsersPool();
        } catch (PapillonBusinessException e) {
            throw new ApplicationException("Problem when initializing the XML parsers pool. Quitting.", e);
        }

        // Get current Layout Setting.
        String priorityPackage = null; 
        try {
            priorityPackage =  Enhydra.getApplication().getConfig().getString(PRIORITY_PACKAGE_CONFIG);
            presentationPriorityPackage = priorityPackage;
        } catch (ConfigException e) {
            // nothing... failing to default...
        }

        // Get current Layout Setting.
        String configLayoutClassName = null; 
        try {
            configLayoutClassName =  Enhydra.getApplication().getConfig().getString(LAYOUT_CONFIG);
            layoutClassName = configLayoutClassName;
        } catch (ConfigException e) {
            // nothing... failing to default...
        }
        
        // Get Login Cookie Name.
        String configLoginCookieName = null;
        try {
            configLoginCookieName =  Enhydra.getApplication().getConfig().getString(COOKIE_CONFIG);
            loginCookieName = configLoginCookieName;
        } catch (ConfigException e) {
            // nothing... failing to default...
        }

        // Get pageExpireTime.
        int configPageExpireTime = 0;
        try {
            configPageExpireTime =  Enhydra.getApplication().getConfig().getInt(PAGE_EXPIRE_TIME);
            pageExpireTime = configPageExpireTime;
        } catch (ConfigException e) {
            // nothing... failing to default...
        }
		
		try {
			applicationPrefix = Enhydra.getApplication().getConfig().getString("Application.Prefix");
		}
		catch (com.lutris.util.ConfigException ce) {
				PapillonLogger.writeDebugMsg("no Application.Prefix var in the config file!");
		}
		
		try {
			Papillon.initializeAllCaches();
        } catch (PapillonBusinessException e) {
            ;
        }
		
        // Upgrade the DB layer to a correct version
        // Problem is: the cache initialization has to be done in order to get all available volumes
        // hence, layer modifications should not break the cache mecanism...
        PapillonLogger.writeInfoMsg("Database uses Data Layer Version n°: " + DataLayerVersion.getDBVersion());
        boolean upgraded = false;
		try {
            upgraded = DataLayerVersion.upgradeDB();
        } catch (PapillonBusinessException e) {
            throw new ApplicationException("Could not upgrade DB Layer.", e);
        }
		if (upgraded) {
			try {
				Papillon.initializeAllCaches();
			} catch (PapillonBusinessException e) {
				throw new ApplicationException("Initialize caches error", e);
			}
		}
    }

    public synchronized static void initializeAllCaches() throws PapillonBusinessException {
        // Initialize dictionarie cache

        DictionariesFactory.initializeDictionaryCache();

        // Initialize volume cache
        VolumesFactory.initializeVolumeCache();

        // Initialize volume cache
        VolumeEntriesFactory.initializeEntryCache();

		// Initialize transformer factory
        // FIXME : For Xalan 2_7_0
        XslSheetFactory.initializeTransformerFactory();

        // Initialize xsl sheet cache
        XslSheetFactory.initializeXslSheetCache();

        // Initialize jibiki xsl sheets
        XslSheetFactory.initializeJibikiXslSheet();

        AvailableLanguages.resetCache();
		
    }

    public String getPriorityPackage() {
        return presentationPriorityPackage;
    }
    
    public String getLayoutClassName() {
        return layoutClassName;
    }
    
    public String getLoginCookieName() {
        return loginCookieName;
    }
	
	public String getUrl() {
		return defaultUrl;
	}
    
    public int getPageExpireTime() {
        return pageExpireTime;
    }
    
    public void setPageExpireTime(int time) {
        pageExpireTime = time;
    }
    
    public String getApplicationPrefix() {
        return applicationPrefix;
    }
    
    public String getApplicationName() {
        return appName;
    }
    
    public boolean requestPreprocessor(HttpPresentationComms comms)
    throws Exception {
        return super.requestPreprocessor(comms);
    }
    
    /**
    * This is an optional function, used only by the Multiserver's graphical
     * administration. This bit of HTML appears in the status page for this
     * application. You could add extra status info, for example
     * a list of currently logged in users.
     *
     * @return HTML that is displayed in the status page of the Multiserver.
     */
    public String toHtml() {
        return "Application <I>Papillon</I>, GETA-CLIPS IMAG";
    }
}
