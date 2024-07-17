package com.spring.biz.ex;

import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.GenericXmlApplicationContext;

public class CarUser {
	public static void main(String[] args) {
		AbstractApplicationContext factory = new GenericXmlApplicationContext("applicationContext2.xml");
		
		Car car = (Car)factory.getBean("hcar");
		car.powerOn();
		car.powerOff();
		car.go();
		car.stop();
		car.print();
	}
}
