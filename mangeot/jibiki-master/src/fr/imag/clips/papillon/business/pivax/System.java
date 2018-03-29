/**
 * @author nguyenhong-thai
 * MT System
 */
package fr.imag.clips.papillon.business.pivax;

import fr.imag.clips.papillon.business.PapillonLogger;
import fr.imag.clips.papillon.business.user.User;

/**
 * @author nguyenhong-thai
 *
 */
public class System {
	private String name;
	private String description;
	private String responsability;
	private String organization;
	private String copyright;
	private String adress;
	private String url_link;
	private String handlerClass;
	
	public System(){
		
	}
	
	//Name
	public void setName(String name){
		this.name = name;
	}
	
	public String getName(){
		return this.name;
	}
	
	//Copyright
	public void setCopyright(String copyright){
		this.copyright = copyright;
	}
	
	public String getCopyright() {
		return this.copyright;
	}
	
	//Description
	public void setDescription(String description){
		this.description = description;
	}
	
	public String getDescription(){
		return this.description;
	}
	
	//Responsability
	public void setResponsability(String responsability){
		this.responsability = responsability;
	}
	
	public String getResponsability(){
		return this.responsability;
	}
	
	//Organization
	public void setOrganization(String organization){
		this.organization = organization;
	}
	
	public String getOrganization(){
		return this.organization;
	}
	
	//Adress
	public void setAdress(String adress){
		this.adress = adress;
	}
	
	public String getAdress(){
		return this.adress;
	}
	
	//URL
	public void setUrl_link(String url_link){
		this.url_link = url_link;
	}
	
	public String getUrl_link(){
		return this.url_link;
	}
	
	//Handler
	public void setHandlerClass(String handlerClass){
		this.handlerClass = handlerClass;
	}
	
	public String getHandlerClass(){
		return this.handlerClass;
	}
}
