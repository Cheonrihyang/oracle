package com.spring.biz.common;

import org.aspectj.lang.JoinPoint;

public class AfterThrowingAdvice {
	
	public void exceptionLog(JoinPoint jp, Exception exception) {
		System.out.println("[예외처리] "+exception.getMessage());
	}
}
