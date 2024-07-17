package com.spring.view.board;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.mvc.Controller;

import com.spring.biz.board.BoardVO;
import com.spring.biz.board.impl.BoardDAO;
import com.spring.biz.board.impl.BoardServiceImpl;

public class InsertBoardController implements Controller{

	BoardServiceImpl service;
	public void setBoardService(BoardServiceImpl service) {
		this.service = service;
	}
	@Override
	public ModelAndView handleRequest(HttpServletRequest request, HttpServletResponse response) throws Exception {
		System.out.println("insertBoard.do 성공");
		
		String title = request.getParameter("title");
		String writer = request.getParameter("writer");
		String content = request.getParameter("content");
		ModelAndView mav = new ModelAndView();
		if(title != null) {
			BoardVO boardVO = new BoardVO();
			boardVO.setTitle(title);
			boardVO.setWriter(writer);
			boardVO.setContent(content);
			
			//BoardDAO boardDAO = new BoardDAO();
			service.insertBoard(boardVO);
			
			mav.setViewName("redirect:getBoardList.do");
		}else {
			mav.setViewName("board1/insertBoard");
		}
		
		
		
		return mav;
	}

}
