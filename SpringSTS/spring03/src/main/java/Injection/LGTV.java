package Injection;

import java.util.Enumeration;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

public class LGTV implements TV{
	
	Speaker speaker;
	String[] buyer;
	Map<String, String> addressMap;
	Properties address;
	
	public LGTV() {}
	public LGTV(Speaker speaker) {
		this.speaker = speaker;
	}
	
	//set
	public void setAddressMap(Map<String, String> addressMap) {
		this.addressMap = addressMap;
	}
	//get
	public Map<String, String> getAddressMap(){
		
		return addressMap;
	}
	
	//Properties 파일
	public void setAddress(Properties address) {
		this.address = address;
	}
	
	
	
	public void setSpeaker(Speaker speaker) {
		this.speaker = speaker;
	}
	
	
	
	
	
	
	@Override
	public void powerOn() {
		System.out.println("LGTV.... 전원을 켠다");
	}
	@Override
	public void powerOff() {
		System.out.println("LGTV.... 전원을 끈다");
		
	}
	@Override
	public void volumeUp() {
		speaker.volumeUp();
	}
	@Override
	public void volumeDown() {
		speaker.volumeDown();
	}
	@Override
	public void print() {
		System.out.println("LGTV입니다");
		//Map
		Set<String> keyset = addressMap.keySet();
		Iterator<String> it = keyset.iterator();
		while(it.hasNext()) {
			String key = it.next();
			String value = addressMap.get(key);
			System.out.println(key+": "+value);
		}
		
		//Properties
		Enumeration keys = (Enumeration)address.keys();
		while(keys.hasMoreElements()) {
			String key = (String)keys.nextElement();
			String value = address.getProperty(key);
			System.out.println(key+" = "+value);
		}
	}
}
