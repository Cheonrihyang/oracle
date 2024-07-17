package com.spring.biz.injection;

import org.springframework.stereotype.Component;

@Component("samsungTV")
public class SamsungTV implements TV{
	
	Speaker speaker;
	
	//기본생성자
	public SamsungTV() {}
	
	
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

}
