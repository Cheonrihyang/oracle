package com.spring.biz.injection;

import org.springframework.stereotype.Component;

@Component("apple")
public class AppleSpeaker implements Speaker{
	
	
	@Override
	public void volumeUp() {
		System.out.println("애플스피커---볼륨을 올린다");
	}

	@Override
	public void volumeDown() {
		System.out.println("애플스피커---볼륨을 내린다");
	}

}
