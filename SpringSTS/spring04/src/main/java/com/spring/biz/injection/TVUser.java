package com.spring.biz.injection;

import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.GenericXmlApplicationContext;


public class TVUser {
	public static void main(String[] args) {
		
		AbstractApplicationContext factory = new GenericXmlApplicationContext("applicationContext.xml");
		TV tv = (TV)factory.getBean("lgTV");
		tv.powerOn();
		tv.powerOff();
		tv.volumeUp();
		tv.volumeDown();	
		
	}
}
