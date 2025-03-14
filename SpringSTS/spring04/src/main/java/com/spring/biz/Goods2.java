package com.spring.biz;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import com.spring.biz.injection.Speaker;
import com.spring.biz.injection.TV;

@Component("goods2")
public class Goods2 implements TV{

	@Autowired
	@Qualifier("apple")
	Speaker speaker;
	
	@Override
	public void powerOn() {
		System.out.println("제품2의 전원을 켠다");
	}

	@Override
	public void powerOff() {
		System.out.println("제품2의 전원을 끈다");
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
