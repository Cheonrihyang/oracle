package com.spring.biz.injection;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

@Component("goods1")
public class Goods1 implements TV{

	@Autowired
	@Qualifier("apple")
	Speaker speaker;
	
	@Override
	public void powerOn() {
		System.out.println("제품1의 전원을 켠다");
	}

	@Override
	public void powerOff() {
		System.out.println("제품1의 전원을 끈다");
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
