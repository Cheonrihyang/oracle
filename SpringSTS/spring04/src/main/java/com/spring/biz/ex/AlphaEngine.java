package com.spring.biz.ex;

import org.springframework.stereotype.Component;

@Component("alpha")
public class AlphaEngine implements Engine{

	@Override
	public void powerOn() {
		System.out.println("알파엔진의 엔진이 켜졌습니다.");
	}

	@Override
	public void powerOff() {
		System.out.println("알파엔진의 엔진이 꺼졌습니다");
	}
	
}
