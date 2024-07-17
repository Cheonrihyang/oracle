package com.spring.biz.injection;

import org.springframework.stereotype.Component;

@Component("sony")
public class SonySpeaker implements Speaker{
	

	@Override
	public void volumeUp() {
		System.out.println("소니스피커---볼륨을 올린다");
	}

	@Override
	public void volumeDown() {
		System.out.println("소니스피커---볼륨을 내린다");
	}
	
}
