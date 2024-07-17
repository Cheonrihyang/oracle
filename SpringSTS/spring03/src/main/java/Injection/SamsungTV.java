package Injection;

import java.util.List;

public class SamsungTV implements TV{
	
	Speaker speaker;
	int price;
	String spec;
	List<String> buyer;
	String[] address;
	
	public SamsungTV() {}
	public SamsungTV(Speaker speaker) {
		this.speaker = speaker;
	}
	public SamsungTV(List<String> buyer, String[] address) {
		this.buyer = buyer;
		this.address = address;
	}

	public SamsungTV(Speaker speaker, int price, String spec) {
		this.speaker = speaker;
		this.price = price;
		this.spec = spec;
	}
	
	public void setSpeaker(Speaker speaker) {
		this.speaker = speaker;
	}
	
	public void setPrice(int price) {
		this.price = price;
	}
	
	public void setSpec(String spec) {
		this.spec = spec;
	}
	
	@Override
	public void powerOn() {
		System.out.println("SamsungTV....전원을 켠다");
	}
	@Override
	public void powerOff() {
		System.out.println("SamsungTV.... 전원을 끈다");
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
		System.out.println("삼성TV: "+spec+", "+price+"원");
		for(String str : buyer) {
			System.out.println(str);
		}
		for(String str1 : address) {
			System.out.println(str1);
		}
	}
}
