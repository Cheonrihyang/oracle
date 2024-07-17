package com.spring.biz.common;

import org.aspectj.lang.JoinPoint;

import com.spring.biz.user.UserVO;

public class AfterReturnAdvice {

	public void afterLog(JoinPoint jp, Object returnObj) {
		
		String method = jp.getSignature().getName();
		if(returnObj instanceof UserVO) {
			UserVO user = (UserVO)returnObj;
			if(user.getId().equals("admin")) {
				System.out.println(user.getName()+" 로그인 하셨습니다.");
			}
		}
		System.out.println("[사후처리]"+method+"()메서드 리턴값: "+returnObj.toString());
	}
}
