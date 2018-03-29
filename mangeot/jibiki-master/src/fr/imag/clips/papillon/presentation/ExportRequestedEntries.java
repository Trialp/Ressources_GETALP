package fr.imag.clips.papillon.presentation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Date;
import java.util.Iterator;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.w3c.dom.Node;

import com.lutris.appserver.server.httpPresentation.HttpPresentationException;
import com.lutris.appserver.server.httpPresentation.HttpPresentationOutputStream;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.dictionary.IQuery;
import fr.imag.clips.papillon.business.dictionary.Volume;
import fr.imag.clips.papillon.business.dictionary.VolumeEntriesFactory;
import fr.imag.clips.papillon.business.dictionary.VolumeEntry;
import fr.imag.clips.papillon.business.dictionary.VolumesFactory;

public class ExportRequestedEntries extends DownloadBasePO {

	public void dumpRequestedData(HttpPresentationOutputStream out)
			throws HttpPresentationException, IOException, Exception {
		// Initialize
        java.util.Vector myKeys = new java.util.Vector();
        java.util.Vector myClauses = new java.util.Vector();
        Volume myVolume = VolumesFactory.getVolumeByName("*ALL*");

        String[] CNFS = new String[4];
        CNFS[0] = Volume.CDM_contributionStatus;
        CNFS[1] = Volume.DEFAULT_LANG;
        CNFS[2] = VolumeEntry.FINISHED_STATUS;
        CNFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];
        myKeys.add(CNFS);

        String[] CFS = new String[4];
        CFS[0] = Volume.CDM_contributionStatus;
        CFS[1] = Volume.DEFAULT_LANG;
        CFS[2] = VolumeEntry.MODIFIED_STATUS;
        CFS[3] = IQuery.QueryBuilderStrategy[IQuery.STRATEGY_EQUAL+1];	
        myKeys.add(CFS);
        
        exportVolumes(myKeys, myClauses, false, out);

	}

	public String getContentType() {
		return "content-type: application/zip";
	}

	public String getFilename() {
		Date now = new Date();
		DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH_mm_ss");
		String dat = dateFormat.format(now);

		return "Lexalp-" + dat + ".zip";
	}

	public void prepareRequest() throws HttpPresentationException, IOException,
			Exception {
		// TODO Auto-generated method stub

	}

	public Node getDocument() throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	protected boolean loggedInUserRequired() {
		return true;
	}

	protected boolean userMayUseThisPO() {
		return true;
	}

	protected void exportVolumes(java.util.Vector myKeys, java.util.Vector myClauses, boolean andClauses, HttpPresentationOutputStream out)
		throws fr.imag.clips.papillon.business.PapillonBusinessException,
		java.io.IOException,
		PapillonPresentationException {

		ZipOutputStream zout = new ZipOutputStream(out);
		for (Iterator iter = ((Collection)VolumesFactory.getVolumes()).iterator(); iter.hasNext();) {
			Volume vol = (Volume)iter.next();
			String filename = vol.getName() + ".xml";

			// -> XML
			zout.putNextEntry(new ZipEntry(filename));
			VolumeEntriesFactory.exportVolume(vol.getName(), myKeys, myClauses, andClauses, VolumeEntriesFactory.XMLFormat, zout);
			zout.closeEntry();
			//
			String userMessage = "Volume " + vol.getName() + " exported";

			//
			if (userMessage != null) {
				this.getSessionData().writeUserMessage(userMessage);
				PapillonLogger.writeDebugMsg(userMessage);
			}
		}
		zout.flush();
		zout.finish();

	}
}
