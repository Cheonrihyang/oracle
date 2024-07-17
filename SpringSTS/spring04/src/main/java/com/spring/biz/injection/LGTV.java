package com.spring.biz.injection;

import javax.annotation.Resource;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

@Component("lgTV")
public class LGTV implements TV{
	//@Autowired
	//@Qualifier("sony")
	@Resource(name="apple")
	Speaker speaker;
	
	//기본생성자
	public LGTV() {}
	
	
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
	
}
